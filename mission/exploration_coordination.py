"""Team-level frontier reconciliation, exhaustion, homing, and graph upkeep."""

from __future__ import annotations

from dataclasses import dataclass
import threading
from typing import Any, Callable, Sequence

from navigation.waypoint_graph import graph_delta_trace_fields


@dataclass(frozen=True)
class FrontierReconciliation:
    """One detached result of applying canonical registry state to the team."""

    active_cluster_ids: tuple[int, ...] = ()
    retired_cluster_ids: tuple[int, ...] = ()
    released_assignment_tokens: tuple[int, ...] = ()
    graph_revisions: tuple[int, ...] = ()


class TeamExplorationCoordinator:
    """Require a coherent team frontier epoch before coordinated homing."""

    def __init__(
        self,
        *,
        registry: Any,
        assignments: Any,
        get_drones: Callable[[], Sequence[Any]],
        gateway_manager: Any | None = None,
        waypoint_graph: Any | None = None,
        runtime_trace: Any | None = None,
        graph_maintenance_interval: int = 32,
    ) -> None:
        self.registry = registry
        self.assignments = assignments
        self.get_drones = get_drones
        self.gateway_manager = gateway_manager
        self.waypoint_graph = waypoint_graph
        self.runtime_trace = runtime_trace
        self.graph_maintenance_interval = max(1, int(graph_maintenance_interval))
        self._lock = threading.RLock()
        self._reported_exhausted: set[int] = set()
        self._team_exhausted = False
        self._last_maintained_revision = (
            0 if waypoint_graph is None else waypoint_graph.topology_revision
        )

    @property
    def team_exhausted(self) -> bool:
        with self._lock:
            return self._team_exhausted

    def synchronize(self) -> FrontierReconciliation:
        """Reconcile runtime views, reservations, gateways, and safe graph nodes."""
        with self._lock:
            drones = tuple(self.get_drones())
            canonical = tuple(self.registry.canonical_clusters())
            active = tuple(sorted(
                cluster.id for cluster in canonical
                if cluster.lifecycle != "retired" and cluster.known_by
            ))
            active_set = set(active)
            if active_set and not self._team_exhausted:
                # Any frontier starts a new team epoch. Earlier empty-view
                # reports cannot be reused after another drone discovers work.
                self._reported_exhausted.clear()
            removed: set[int] = set()
            intents = []
            for drone in drones:
                visible = self.registry.visible_to(drone.id)
                removed.update(
                    drone.runtime_state.reconcile_frontier_clusters(visible)
                )
                if visible:
                    self._reported_exhausted.discard(int(drone.id))
                intents.append(drone.runtime_state.navigation_intent())

            released = self.assignments.reconcile_active_clusters(active_set)
            active_route_nodes: set[int] = set()
            active_route_edges: set[int] = set()
            for intent in intents:
                if intent is None:
                    continue
                active_route_nodes.update(intent.route_node_ids)
                active_route_edges.update(intent.route_edge_ids)
                active_route_edges.update(
                    edge_id for edge_id in intent.route_segment_edge_ids
                    if edge_id is not None
                )

            graph_revisions: list[int] = []
            graph_deltas = []
            active_gateway_ids = {
                int(cluster.gateway_id) for cluster in canonical
                if cluster.id in active_set and cluster.gateway_id is not None
            }
            if self.gateway_manager is not None:
                for cluster in canonical:
                    if (
                        cluster.lifecycle != "retired"
                        or cluster.gateway_id is None
                        or cluster.gateway_id in active_gateway_ids
                    ):
                        continue
                    delta = self.gateway_manager.retire_gateway(
                        cluster.id,
                        active_route_node_ids=active_route_nodes,
                        active_route_edge_ids=active_route_edges,
                    )
                    if delta.revision and (
                        delta.retired_node_ids
                        or delta.retired_edge_ids
                        or delta.updated_node_ids
                    ):
                        graph_revisions.append(delta.revision)
                        graph_deltas.append(delta)

            graph = self.waypoint_graph
            if graph is not None and (
                removed
                or graph.topology_revision - self._last_maintained_revision
                >= self.graph_maintenance_interval
            ):
                leaf_deltas = graph.retire_inactive_orphan_trail_leaves(
                    active_route_node_ids=active_route_nodes,
                    active_route_edge_ids=active_route_edges,
                    active_gateway_ids=active_gateway_ids,
                )
                near_deltas = graph.collapse_inactive_nearby_nodes(
                    active_route_node_ids=active_route_nodes,
                    active_route_edge_ids=active_route_edges,
                    active_gateway_ids=active_gateway_ids,
                    radius=8.0,
                )
                deltas = (
                    leaf_deltas
                    + near_deltas
                    + graph.collapse_inactive_degree_two_nodes(
                        active_route_node_ids=active_route_nodes,
                        active_route_edge_ids=active_route_edges,
                        active_gateway_ids=active_gateway_ids,
                    )
                )
                graph_revisions.extend(delta.revision for delta in deltas)
                graph_deltas.extend(deltas)
                self._last_maintained_revision = graph.topology_revision

            result = FrontierReconciliation(
                active_cluster_ids=active,
                retired_cluster_ids=tuple(sorted(removed)),
                released_assignment_tokens=tuple(sorted(
                    assignment.token for assignment in released
                )),
                graph_revisions=tuple(graph_revisions),
            )
            if (
                self.runtime_trace is not None
                and (
                    result.retired_cluster_ids
                    or result.released_assignment_tokens
                    or result.graph_revisions
                )
            ):
                for delta in graph_deltas:
                    self.runtime_trace.record(
                        "waypoint_graph_delta",
                        **graph_delta_trace_fields(graph, delta),
                    )
                self.runtime_trace.record(
                    "team_frontiers_reconciled",
                    active_cluster_ids=result.active_cluster_ids,
                    retired_cluster_ids=result.retired_cluster_ids,
                    released_assignment_tokens=(
                        result.released_assignment_tokens
                    ),
                    graph_revisions=result.graph_revisions,
                )
            return result

    def note_frontier_refresh(self, drone_id: int) -> FrontierReconciliation:
        """Invalidate one exhaustion report when a fresh local frontier exists."""
        result = self.synchronize()
        if self.registry.visible_to(int(drone_id)):
            with self._lock:
                self._reported_exhausted.discard(int(drone_id))
        return result

    def refresh_frontiers(
        self,
        drone_id: int,
        components: Sequence[Any],
        *,
        slam_version: int,
    ) -> tuple[Any, ...]:
        """Serialize canonical refreshes with the team exhaustion handshake."""
        with self._lock:
            clusters = self.registry.refresh(
                int(drone_id),
                components,
                slam_version=int(slam_version),
            )
            self.synchronize()
            return clusters

    def retire_cluster(self, cluster_id: int, *, reason: str) -> bool:
        """Serialize explicit retirement and immediately reconcile the team."""
        with self._lock:
            retired = self.registry.retire(int(cluster_id), reason=str(reason))
            if retired:
                self.synchronize()
            return retired

    def report_exhausted(self, drone_id: int) -> bool:
        """Latch team homing only after every drone reports a frontier-free view."""
        with self._lock:
            if self._team_exhausted:
                return True
            result = self.synchronize()
            observer = int(drone_id)
            if result.active_cluster_ids or self.registry.visible_to(observer):
                self._reported_exhausted.discard(observer)
                return False
            drones = tuple(self.get_drones())
            self._reported_exhausted.add(observer)
            participants = {int(drone.id) for drone in drones}
            if not participants or not participants.issubset(self._reported_exhausted):
                return False

            self._team_exhausted = True
            for drone in drones:
                drone.runtime_state.start_returning_home()
            if self.runtime_trace is not None:
                self.runtime_trace.record(
                    "team_exploration_exhausted",
                    drone_ids=tuple(sorted(participants)),
                    active_cluster_ids=(),
                )
            return True
