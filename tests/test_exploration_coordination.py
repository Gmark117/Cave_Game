import unittest
from types import SimpleNamespace

import numpy as np

from agents.drone_runtime_state import DroneRuntimeState
from mission.exploration_coordination import TeamExplorationCoordinator
from navigation.frontier_clusters import (
    AssignmentRegistry,
    FrontierClusterRegistry,
    FrontierComponent,
)
from navigation.waypoint_graph import WaypointGraph, WaypointRole


def component(position: tuple[int, int]) -> FrontierComponent:
    x, y = position
    return FrontierComponent(
        cells=frozenset({position}),
        bounds=(x, y, x + 1, y + 1),
        representative=position,
        expected_gain=2,
    )


def drone(drone_id: int):
    state = DroneRuntimeState(
        start_position=(1, 1),
        cave=np.zeros((8, 8), dtype=np.uint8),
        direction=0,
        frontier_rebuild_cooldown=0.0,
    )
    return SimpleNamespace(id=drone_id, runtime_state=state, snapshot=state.snapshot)


class RecordingTrace:
    def __init__(self) -> None:
        self.events = []

    def record(self, event: str, **fields) -> None:
        self.events.append((event, fields))


class TeamExplorationCoordinatorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.registry = FrontierClusterRegistry(missing_refresh_limit=0)
        self.assignments = AssignmentRegistry()
        self.drones = [drone(0), drone(1), drone(2)]
        self.coordinator = TeamExplorationCoordinator(
            registry=self.registry,
            assignments=self.assignments,
            get_drones=lambda: self.drones,
        )

    def test_retirement_reconciles_every_runtime_snapshot_and_assignment(self) -> None:
        cluster = self.registry.refresh(
            0, (component((4, 4)),), slam_version=1
        )[0]
        self.registry.share(0, 1)
        self.registry.share(0, 2)
        for item in self.drones:
            item.runtime_state.replace_frontier_clusters(
                self.registry.visible_to(item.id)
            )
        assignment = self.assignments.reserve(cluster_id=cluster.id, drone_id=1)

        self.registry.retire(cluster.id, reason="resolved_by_peer")
        reconciliation = self.coordinator.synchronize()

        self.assertEqual(reconciliation.retired_cluster_ids, (cluster.id,))
        self.assertTrue(all(not item.snapshot().frontier_cluster_ids for item in self.drones))
        self.assertIsNone(self.assignments.assignment_for_token(assignment.token))

    def test_homing_latches_only_after_every_drone_confirms_team_exhaustion(self) -> None:
        self.assertFalse(self.coordinator.report_exhausted(0))
        self.assertFalse(self.coordinator.report_exhausted(1))
        self.assertTrue(all(not item.snapshot().returning_home for item in self.drones))

        self.assertTrue(self.coordinator.report_exhausted(2))

        self.assertTrue(self.coordinator.team_exhausted)
        self.assertTrue(all(item.snapshot().returning_home for item in self.drones))

    def test_any_canonical_frontier_blocks_team_homing(self) -> None:
        self.registry.refresh(2, (component((4, 4)),), slam_version=1)

        self.coordinator.report_exhausted(0)
        self.coordinator.report_exhausted(1)
        exhausted = self.coordinator.report_exhausted(2)

        self.assertFalse(exhausted)
        self.assertFalse(self.coordinator.team_exhausted)
        self.assertTrue(all(not item.snapshot().returning_home for item in self.drones))

    def test_new_frontier_resets_every_prior_exhaustion_report(self) -> None:
        self.assertFalse(self.coordinator.report_exhausted(0))
        cluster = self.registry.refresh(
            1, (component((4, 4)),), slam_version=1
        )[0]
        self.coordinator.note_frontier_refresh(1)
        self.registry.retire(cluster.id, reason="resolved")
        self.coordinator.synchronize()

        self.assertFalse(self.coordinator.report_exhausted(1))
        self.assertFalse(self.coordinator.report_exhausted(2))
        self.assertTrue(self.coordinator.report_exhausted(0))

    def test_graph_maintenance_emits_canonical_retirement_delta(self) -> None:
        graph = WaypointGraph(merge_radius=0)
        trace = RecordingTrace()
        coordinator = TeamExplorationCoordinator(
            registry=self.registry,
            assignments=self.assignments,
            get_drones=lambda: self.drones,
            waypoint_graph=graph,
            runtime_trace=trace,
            graph_maintenance_interval=1,
        )
        graph.register_travelled_section(
            ((0, 0), (4, 0)), end_role=WaypointRole.TURN,
        )
        graph.register_travelled_section(((4, 0), (8, 0)))

        coordinator.synchronize()

        delta = next(
            fields for event, fields in trace.events
            if event == "waypoint_graph_delta"
        )
        self.assertTrue(delta["retired_node_ids"])
        self.assertTrue(delta["retired_edge_ids"])
        self.assertIn("node_count", delta)


if __name__ == "__main__":
    unittest.main()
