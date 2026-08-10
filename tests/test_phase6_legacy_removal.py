"""Phase 6 characterization guards for obsolete planner-facing APIs."""

from __future__ import annotations

from dataclasses import fields
import inspect
import unittest

from agents.drone import Drone
from agents.drone_movement import DroneMovementController
from agents.drone_runtime_state import DroneRuntimeState
from agents.exploration_policy import ExplorationContext, FrontierExplorationPolicy
from config.simulation_config import ExplorationConfig, FrontierConfig, WaypointConfig
from contracts import DroneMovementDependencies
from mission.control import MissionControl
from navigation.waypoint_graph import WaypointGraph


class Phase6LegacyRemovalTests(unittest.TestCase):
    def test_planner_context_and_drone_dependencies_expose_no_ground_truth(self) -> None:
        context_fields = {field.name for field in fields(ExplorationContext)}
        dependency_fields = {
            field.name for field in fields(DroneMovementDependencies)
        }

        self.assertNotIn("cave_map", context_fields)
        self.assertNotIn("compute_path", dependency_fields)

    def test_runtime_has_no_raw_frontier_or_ground_truth_astar_fallbacks(self) -> None:
        obsolete_methods = {
            FrontierExplorationPolicy: {
                "choose_next_step",
                "extract_frontiers",
                "prioritize_frontiers",
                "frontier_distance",
            },
            DroneMovementController: {
                "find_new_node",
                "explore",
                "reach_border",
                "get_distance",
                "_compute_path",
                "_direct_path_limit",
            },
            DroneRuntimeState: {
                "replace_frontiers",
                "merge_frontiers",
                "frontier_cluster_id",
            },
            Drone: {"merge_frontiers"},
            MissionControl: {"compute_path"},
        }

        for owner, names in obsolete_methods.items():
            for name in names:
                with self.subTest(owner=owner.__name__, name=name):
                    self.assertFalse(hasattr(owner, name))

    def test_breadcrumb_sampler_and_legacy_typed_settings_are_removed(self) -> None:
        self.assertFalse(hasattr(WaypointGraph, "register_travelled_path"))
        self.assertNotIn("spacing", inspect.signature(WaypointGraph).parameters)

        frontier_fields = {field.name for field in fields(FrontierConfig)}
        waypoint_fields = {field.name for field in fields(WaypointConfig)}
        exploration_fields = {field.name for field in fields(ExplorationConfig)}

        self.assertNotIn("stride", frontier_fields)
        self.assertNotIn("spacing", waypoint_fields)
        self.assertNotIn("direct_path_limit", waypoint_fields)
        self.assertNotIn("frontier_cluster_limit", exploration_fields)


if __name__ == "__main__":
    unittest.main()
