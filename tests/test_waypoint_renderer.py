import os
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

import pygame

from navigation.waypoint_graph import EDGE_SLAM_LOS, EDGE_TRAVELLED
from rendering.waypoint_renderer import WaypointRenderer


class RecordingGraph:
    """Minimal versioned graph used to verify renderer cache behavior."""

    def __init__(self, snapshot) -> None:
        self.version = snapshot.version
        self.current_snapshot = snapshot
        self.snapshot_calls = 0

    def snapshot(self):
        self.snapshot_calls += 1
        return self.current_snapshot

    def replace(self, snapshot) -> None:
        self.current_snapshot = snapshot
        self.version = snapshot.version


def graph_snapshot(version, *, edges=(), waypoints=()):
    return SimpleNamespace(
        version=version,
        edges=tuple(edges),
        waypoints=tuple(waypoints),
    )


class WaypointRendererTests(unittest.TestCase):
    def test_draws_edges_before_source_distinguished_nodes(self) -> None:
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
        waypoints = (
            SimpleNamespace(position=(2, 4), source="home"),
            SimpleNamespace(position=(8, 4), source=EDGE_TRAVELLED),
            SimpleNamespace(position=(14, 4), source="gateway"),
        )
        graph = RecordingGraph(
            graph_snapshot(3, edges=edges, waypoints=waypoints)
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

    def test_reuses_surface_until_graph_version_changes(self) -> None:
        graph = RecordingGraph(graph_snapshot(0))
        renderer = WaypointRenderer(graph, 16, 10)
        target = Mock()

        renderer.draw(target)
        renderer.draw(target)

        self.assertEqual(graph.snapshot_calls, 1)
        self.assertEqual(renderer.rendered_version, 0)

        graph.replace(
            graph_snapshot(
                1,
                waypoints=(
                    SimpleNamespace(position=(4, 4), source="home"),
                ),
            )
        )
        renderer.draw(target)
        renderer.draw(target)

        self.assertEqual(graph.snapshot_calls, 2)
        self.assertEqual(renderer.rendered_version, 1)
        self.assertEqual(target.blit.call_count, 4)
        self.assertEqual(renderer.surface.get_at((4, 4)), (80, 140, 255, 245))


if __name__ == "__main__":
    unittest.main()
