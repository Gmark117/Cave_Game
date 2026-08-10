import os
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

import pygame

from navigation.waypoint_graph import (
    EDGE_SLAM_LOS,
    EDGE_TRAVELLED,
    WaypointGraph,
    WaypointRole,
)
from rendering.waypoint_renderer import WaypointRenderer


class RecordingGraph:
    """Minimal revisioned graph used to verify renderer cache behavior."""

    def __init__(self, snapshot) -> None:
        self.topology_revision = snapshot.topology_revision
        self.current_snapshot = snapshot
        self.snapshot_calls = 0

    def snapshot(self):
        self.snapshot_calls += 1
        return self.current_snapshot

    def replace(self, snapshot) -> None:
        self.current_snapshot = snapshot
        self.topology_revision = snapshot.topology_revision


def graph_snapshot(topology_revision, *, edges=(), nodes=()):
    return SimpleNamespace(
        topology_revision=topology_revision,
        edges=tuple(edges),
        nodes=tuple(nodes),
    )


class WaypointRendererTests(unittest.TestCase):
    def test_draws_edges_before_strategic_role_nodes(self) -> None:
        edges = (
            SimpleNamespace(
                source=EDGE_TRAVELLED,
                path=((1, 1), (8, 1)),
            ),
            SimpleNamespace(
                source=EDGE_SLAM_LOS,
                path=((8, 1), (8, 8)),
            ),
        )
        nodes = (
            SimpleNamespace(
                position=(2, 4),
                roles=frozenset({WaypointRole.HOME}),
            ),
            SimpleNamespace(
                position=(8, 4),
                roles=frozenset({WaypointRole.TURN}),
            ),
            SimpleNamespace(
                position=(14, 4),
                roles=frozenset({WaypointRole.FRONTIER_GATEWAY}),
            ),
        )
        graph = RecordingGraph(
            graph_snapshot(3, edges=edges, nodes=nodes)
        )
        renderer = WaypointRenderer(graph, 20, 12)
        target = Mock()
        draw_order = []

        with patch(
            "rendering.waypoint_renderer.pygame.draw.lines",
            side_effect=lambda *args, **kwargs: draw_order.append(
                ("edge", args)
            ),
        ), patch(
            "rendering.waypoint_renderer.pygame.draw.circle",
            side_effect=lambda *args, **kwargs: draw_order.append(
                ("node", args)
            ),
        ):
            renderer.draw(target)

        self.assertEqual(renderer.surface.get_size(), (20, 12))
        self.assertEqual(
            [kind for kind, _args in draw_order],
            ["edge", "edge", "node", "node", "node"],
        )
        edge_styles = [(args[1], args[4]) for _, args in draw_order[:2]]
        self.assertEqual(len(set(edge_styles)), 2)
        node_colors = [args[1] for _, args in draw_order[2:]]
        self.assertEqual(len(set(node_colors)), 3)
        target.blit.assert_called_once_with(renderer.surface, (0, 0))

    def test_reuses_surface_until_topology_revision_changes(self) -> None:
        graph = RecordingGraph(graph_snapshot(0))
        renderer = WaypointRenderer(graph, 16, 10)
        target = Mock()

        renderer.draw(target)
        renderer.draw(target)

        self.assertEqual(graph.snapshot_calls, 1)
        self.assertEqual(renderer.rendered_revision, 0)

        graph.replace(
            graph_snapshot(
                1,
                nodes=(
                    SimpleNamespace(
                        position=(4, 4),
                        roles=frozenset({WaypointRole.HOME}),
                    ),
                ),
            )
        )
        renderer.draw(target)
        renderer.draw(target)

        self.assertEqual(graph.snapshot_calls, 2)
        self.assertEqual(renderer.rendered_revision, 1)
        self.assertEqual(target.blit.call_count, 4)
        self.assertEqual(renderer.surface.get_at((4, 4)), (80, 140, 255, 245))

    def test_strategic_role_style_does_not_require_node_source(self) -> None:
        graph = RecordingGraph(graph_snapshot(
            1,
            nodes=(SimpleNamespace(
                id=7,
                position=(5, 5),
                roles=frozenset({WaypointRole.JUNCTION}),
            ),),
        ))
        renderer = WaypointRenderer(graph, 12, 12)

        renderer.draw(Mock())

        self.assertEqual(renderer.surface.get_at((5, 5)), (255, 235, 90, 245))

    def test_every_strategic_role_has_a_distinct_stable_style(self) -> None:
        expected = {
            WaypointRole.HOME: (80, 140, 255, 245),
            WaypointRole.JUNCTION: (255, 235, 90, 245),
            WaypointRole.CHOKEPOINT: (255, 90, 90, 245),
            WaypointRole.TURN: (80, 235, 175, 225),
            WaypointRole.FRONTIER_GATEWAY: (255, 126, 45, 240),
            WaypointRole.RECOVERY_ANCHOR: (210, 210, 255, 225),
        }
        positions = {
            role: (4 + index * 8, 8)
            for index, role in enumerate(expected)
        }
        graph = RecordingGraph(graph_snapshot(
            7,
            nodes=tuple(
                SimpleNamespace(
                    id=index + 1,
                    position=positions[role],
                    roles=frozenset({role}),
                )
                for index, role in enumerate(expected)
            ),
        ))
        renderer = WaypointRenderer(graph, 52, 16)

        renderer.draw(Mock())

        actual = {
            role: tuple(renderer.surface.get_at(position))
            for role, position in positions.items()
        }
        self.assertEqual(actual, expected)
        self.assertEqual(len(set(actual.values())), len(WaypointRole))

    def test_rebuilds_once_for_each_committed_topology_revision(self) -> None:
        graph = WaypointGraph(merge_radius=0)
        renderer = WaypointRenderer(graph, 24, 24)
        target = Mock()

        with patch.object(
            renderer,
            "_rebuild",
            wraps=renderer._rebuild,
        ) as rebuild:
            renderer.draw(target)
            renderer.draw(target)
            self.assertEqual(rebuild.call_count, 1)

            first = graph.register_travelled_section(
                ((2, 2), (18, 2)),
                start_role=WaypointRole.HOME,
                end_role=WaypointRole.TURN,
            )
            self.assertEqual(first.delta.revision, 1)
            self.assertEqual(graph.topology_revision, 1)
            renderer.draw(target)
            renderer.draw(target)
            self.assertEqual(rebuild.call_count, 2)

            replay = graph.register_travelled_section(
                ((2, 2), (18, 2)),
                start_role=WaypointRole.HOME,
                end_role=WaypointRole.TURN,
            )
            self.assertFalse(replay.changed)
            self.assertEqual(graph.topology_revision, 1)
            renderer.draw(target)
            self.assertEqual(rebuild.call_count, 2)

            second = graph.register_travelled_section(
                ((18, 2), (18, 18)),
                start_role=WaypointRole.JUNCTION,
                end_role=WaypointRole.CHOKEPOINT,
            )
            self.assertEqual(second.delta.revision, 2)
            self.assertEqual(graph.topology_revision, 2)
            renderer.draw(target)
            renderer.draw(target)
            self.assertEqual(rebuild.call_count, 3)

        self.assertEqual(renderer.rendered_revision, 2)


if __name__ == "__main__":
    unittest.main()
