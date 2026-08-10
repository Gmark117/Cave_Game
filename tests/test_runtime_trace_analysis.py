"""Synthetic replay tests for structured runtime-trace characterization."""

from __future__ import annotations

import math
import unittest

from tools.analyze_runtime_trace import (
    LEGACY_MCTS_BUDGET_MS,
    analyze_trace,
    format_characterization,
)


class RuntimeTraceAnalysisTests(unittest.TestCase):
    def test_replacement_trace_reports_every_final_acceptance_signal(self) -> None:
        events = [
            {
                "event": "mission_constructed",
                "schema_version": 3,
                "sequence": 0,
                "mcts_decision_time_budget_ms": 40.0,
            },
            {
                "event": "waypoint_graph_delta",
                "schema_version": 3,
                "sequence": 1,
                "topology_revision": 1,
                "added_nodes": [
                    {
                        "node_id": 1,
                        "position": [0, 0],
                        "roles": ["home"],
                    },
                    {
                        "node_id": 2,
                        "position": [16, 0],
                        "roles": ["turn"],
                    },
                ],
                "added_edges": [
                    {
                        "edge_id": 1,
                        "start_node_id": 1,
                        "end_node_id": 2,
                    }
                ],
                "retired_node_ids": [],
                "retired_edge_ids": [],
                "edge_replacements": {},
            },
            # Legacy aliases deliberately remain during Phase 6 validation.
            {
                "event": "waypoint_added",
                "schema_version": 3,
                "sequence": 2,
                "node_id": 1,
                "waypoint": [0, 0],
                "source": "home",
            },
            {
                "event": "waypoint_edge_added",
                "schema_version": 3,
                "sequence": 3,
                "edge_id": 1,
                "start": [0, 0],
                "end": [16, 0],
            },
            {
                "event": "drone_waypoint_route",
                "schema_version": 3,
                "sequence": 4,
                "drone_id": 0,
                "target": [16, 0],
                "status": "ok",
                "route_elapsed_ms": 1.0,
                "route": {
                    "route_id": 11,
                    "cache_hit": False,
                    "topology_revision": 1,
                    "requester_knowledge_revision": 4,
                    "node_ids": [1, 2],
                    "edge_ids": [1],
                    "total_cost": 16.0,
                    "remaining_cost": 16.0,
                },
                "replan_reason": "selected",
                "route_replanned": True,
                "active_intent_valid": False,
            },
            {
                "event": "drone_waypoint_route",
                "schema_version": 3,
                "sequence": 5,
                "drone_id": 0,
                "target": [16, 0],
                "status": "ok",
                "route_elapsed_ms": 2.0,
                "route": {
                    "route_id": 12,
                    "cache_hit": True,
                    "topology_revision": 1,
                    "requester_knowledge_revision": 4,
                    "node_ids": [1, 2],
                    "edge_ids": [1],
                    "total_cost": 16.0,
                    "remaining_cost": 16.0,
                },
                "replan_reason": "invalidated",
                "route_replanned": True,
                "active_intent_valid": False,
            },
            {
                "event": "drone_navigation_transition",
                "schema_version": 3,
                "sequence": 6,
                "drone_id": 0,
                "replan_reason": "selected",
                "mode_transition": {
                    "from_mode": None,
                    "to_mode": "travel",
                    "reason": "selected",
                },
                "intent": {
                    "intent_id": 21,
                    "mode": "travel",
                    "goal_cluster_id": 7,
                    "gateway_id": 2,
                    "assignment_id": 5,
                    "route_id": 11,
                    "topology_revision": 1,
                    "requester_knowledge_revision": 4,
                    "edge_cursor": 0,
                    "polyline_cursor": 0,
                    "remaining_route_cost": 16.0,
                },
            },
            {
                "event": "drone_waypoint_segment_path",
                "schema_version": 3,
                "sequence": 7,
                "drone_id": 0,
                "path_source": "persistent_route",
                "astar_path_len": 0,
                "persistent_edge_astar_calls": 0,
                "connector_astar_calls": 1,
            },
            {
                "event": "drone_local_mcts_decision",
                "schema_version": 3,
                "sequence": 8,
                "drone_id": 0,
                "mcts": {
                    "performed": True,
                    "elapsed_ms": 39.0,
                    "iterations": 3,
                    "root_coverage_complete": True,
                    "budget_ms": 40.0,
                },
            },
            {
                "event": "drone_local_mcts_decision",
                "schema_version": 3,
                "sequence": 9,
                "drone_id": 0,
                "mcts": {
                    "performed": True,
                    "elapsed_ms": 41.0,
                    "iterations": 4,
                    "root_coverage_complete": True,
                    "budget_ms": 40.0,
                },
            },
            {
                "event": "drone_watchdog",
                "schema_version": 3,
                "sequence": 10,
                "drone_id": 0,
                "triggered_reason": "reversal",
                "watchdog": {
                    "last_progress_time": 2.0,
                    "distance_without_progress": 0.0,
                    "recent_visits": [1, 2, 1, 2, 1],
                    "reversal_count": 2,
                    "revisit_ratio": 0.6,
                },
            },
            {
                "event": "drone_navigation_transition",
                "schema_version": 3,
                "sequence": 11,
                "drone_id": 0,
                "mode_transition": {
                    "from_mode": "travel",
                    "to_mode": "recovery",
                    "reason": "reversal",
                },
                "intent": {
                    "intent_id": 22,
                    "mode": "recovery",
                    "goal_cluster_id": None,
                    "route_id": 0,
                },
            },
            {
                "event": "drone_motion",
                "schema_version": 3,
                "sequence": 12,
                "drone_id": 0,
                "sim_time": 1.0,
                "travelled_distance": 2.0,
            },
            {
                "event": "sensor_scan",
                "schema_version": 3,
                "sequence": 13,
                "drone_id": 0,
                "sim_time": 1.1,
                "newly_known_cells": 64,
                "confidence_gain": 4.0,
            },
        ]

        metrics = analyze_trace(events)
        acceptance = metrics.acceptance

        self.assertEqual(metrics.mcts.search_count, 2)
        self.assertEqual(metrics.mcts.p99_elapsed_ms, 41.0)
        self.assertEqual(metrics.mcts.incomplete_root_coverage_count, 0)
        self.assertEqual(acceptance.forbidden_node_role_count, 0)
        self.assertEqual(acceptance.unexplained_goal_changes, 0)
        self.assertEqual(acceptance.second_reversal_triggers, 1)
        self.assertEqual(acceptance.second_reversal_recoveries, 1)
        self.assertEqual(acceptance.watchdog_transition_delays, 0)
        self.assertEqual(acceptance.persistent_edge_astar_calls, 0)
        self.assertEqual(acceptance.unchanged_valid_intent_replans, 0)
        self.assertEqual(acceptance.planner_cave_map_field_count, 0)
        self.assertTrue(acceptance.replacement_schema_valid)
        self.assertTrue(acceptance.legacy_trace_fields_present)
        self.assertEqual(
            metrics.information_efficiency.newly_known_cells_per_travelled_px,
            32.0,
        )

        report = "\n".join(format_characterization(metrics))
        for label in (
            "Final acceptance:",
            "strategic graph gates:",
            "goal/watchdog gates:",
            "MCTS gates:",
            "route execution gates:",
            "belief/schema gates:",
        ):
            self.assertIn(label, report)

    def test_schema_audit_detects_missing_fields_and_planner_truth_payloads(
        self,
    ) -> None:
        metrics = analyze_trace([
            {
                "event": "drone_navigation_transition",
                "schema_version": 3,
                "sequence": 0,
                "drone_id": 0,
                "cave_map": [[0]],
                "intent": {"intent_id": 1},
            }
        ])

        self.assertFalse(metrics.acceptance.replacement_schema_valid)
        self.assertGreater(
            metrics.acceptance.replacement_schema_missing_field_count,
            0,
        )
        self.assertEqual(metrics.acceptance.planner_cave_map_field_count, 1)

    def test_waypoint_density_uses_observed_nodes_edges_and_cells(self) -> None:
        events = [
            {
                "event": "waypoint_added",
                "waypoint": [0, 0],
                "source": "home",
                "node_count": 1,
            },
            {
                "event": "waypoint_added",
                "waypoint": [3, 4],
                "source": "travelled",
                "node_count": 2,
            },
            {
                "event": "waypoint_added",
                "waypoint": [32, 0],
                "source": "gateway",
                "node_count": 3,
            },
            {
                "event": "waypoint_added",
                "waypoint": [64, 0],
                "source": "turn",
                "node_count": 4,
            },
            {
                "event": "waypoint_edge_added",
                "start": [0, 0],
                "end": [3, 4],
                "edge_count": 1,
            },
            {
                "event": "waypoint_edge_added",
                "start": [3, 4],
                "end": [32, 0],
                "edge_count": 2,
            },
            {
                "event": "waypoint_edge_added",
                "start": [32, 0],
                "end": [64, 0],
                "edge_count": 3,
            },
        ]

        density = analyze_trace(events).waypoint_density

        self.assertEqual(density.node_count, 4)
        self.assertEqual(density.edge_count, 3)
        self.assertEqual(density.edge_mutation_count, 3)
        self.assertEqual(density.unique_connection_count, 3)
        self.assertEqual(density.source_counts["travelled"], 1)
        self.assertEqual(density.occupied_spatial_cells, 3)
        self.assertAlmostEqual(density.average_nodes_per_occupied_cell, 4 / 3)
        self.assertEqual(density.maximum_nodes_in_cell, 2)
        self.assertEqual(density.nodes_with_neighbor_within_8_px, 2)
        self.assertEqual(density.neighbor_within_8_px_rate, 0.5)
        self.assertEqual(density.nodes_with_neighbor_within_16_px, 2)
        self.assertEqual(density.degree_two_nodes, 2)
        self.assertEqual(density.degree_two_rate, 0.5)
        expected_median = (5.0 + math.dist((3, 4), (32, 0))) / 2
        self.assertAlmostEqual(density.median_nearest_neighbor_px, expected_median)
        self.assertEqual(density.reported_node_count, 4)
        self.assertEqual(density.reported_edge_count, 3)

    def test_density_accepts_id_based_graph_delta_and_retirement(self) -> None:
        events = [
            {
                "event": "waypoint_graph_delta",
                "added_nodes": [
                    {"node_id": 10, "position": [1, 1], "role": "HOME"},
                    {"node_id": 11, "position": [9, 1], "role": "TURN"},
                ],
                "added_edges": [
                    {
                        "edge_id": 20,
                        "start_node_id": 10,
                        "end_node_id": 11,
                    }
                ],
            },
            {
                "event": "waypoint_graph_delta",
                "retired_node_ids": [11],
                "retired_edge_ids": [20],
            },
        ]

        density = analyze_trace(events).waypoint_density

        self.assertEqual(density.node_count, 1)
        self.assertEqual(density.edge_count, 0)
        self.assertEqual(density.edge_mutation_count, 1)
        self.assertEqual(density.unique_connection_count, 0)
        self.assertIsNone(density.median_nearest_neighbor_px)

    def test_density_uses_exclusive_radius_and_simple_topological_degree(
        self,
    ) -> None:
        events = [
            {
                "event": "waypoint_added",
                "waypoint": [0, 0],
                "source": "home",
            },
            {
                "event": "waypoint_added",
                "waypoint": [8, 0],
                "source": "travelled",
            },
            {
                "event": "waypoint_added",
                "waypoint": [16, 0],
                "source": "travelled",
            },
            {
                "event": "waypoint_edge_added",
                "start": [0, 0],
                "end": [8, 0],
                "source": "travelled",
                "edge_count": 1,
            },
            # A shorter same-key mutation replaces the active edge.
            {
                "event": "waypoint_edge_added",
                "start": [0, 0],
                "end": [8, 0],
                "source": "travelled",
                "edge_count": 1,
            },
            # A second evidence source is active in parallel but does not add
            # another topological neighbor.
            {
                "event": "waypoint_edge_added",
                "start": [0, 0],
                "end": [8, 0],
                "source": "slam_los",
                "edge_count": 2,
            },
            {
                "event": "waypoint_edge_added",
                "start": [8, 0],
                "end": [16, 0],
                "source": "travelled",
                "edge_count": 3,
            },
        ]

        density = analyze_trace(events).waypoint_density

        self.assertEqual(density.nodes_with_neighbor_within_8_px, 0)
        self.assertEqual(density.edge_mutation_count, 4)
        self.assertEqual(density.edge_count, 3)
        self.assertEqual(density.reported_edge_count, 3)
        self.assertEqual(density.unique_connection_count, 2)
        self.assertEqual(density.degree_two_nodes, 1)

    def test_density_accepts_updated_replacement_schema_records(self) -> None:
        events = [
            {
                "event": "waypoint_graph_delta",
                "added_nodes": [
                    {"node_id": 1, "position": [0, 0], "role": "HOME"},
                    {"node_id": 2, "position": [4, 0], "role": "TURN"},
                ],
                "added_edges": [
                    {
                        "edge_id": 3,
                        "start_node_id": 1,
                        "end_node_id": 2,
                    }
                ],
            },
            {
                "event": "waypoint_graph_delta",
                "updated_nodes": [
                    {
                        "node_id": 2,
                        "position": [5, 0],
                        "role": ["TURN", "JUNCTION"],
                    }
                ],
                "updated_edges": [
                    {
                        "edge_id": 3,
                        "start_node_id": 1,
                        "end_node_id": 2,
                    }
                ],
            },
        ]

        density = analyze_trace(events).waypoint_density

        self.assertEqual(density.node_count, 2)
        self.assertEqual(density.edge_count, 1)
        self.assertEqual(density.edge_mutation_count, 2)
        self.assertEqual(density.unique_connection_count, 1)
        self.assertEqual(density.median_nearest_neighbor_px, 5.0)

    def test_segment_followups_separate_retention_from_local_diversion(self) -> None:
        target_a = [10, 10]
        target_b = [20, 20]
        events: list[dict[str, object]] = []

        def segment_then(
            decision: dict[str, object],
            post_rebuild: dict[str, object] | None = None,
        ) -> None:
            events.extend(
                [
                    {
                        "event": "drone_waypoint_segment_complete",
                        "drone_id": 0,
                        "target": target_a,
                    },
                    # This closes the action that completed the segment and
                    # must not be mistaken for its next-decision follow-up.
                    {"event": "drone_action_result", "drone_id": 0},
                    {
                        "event": "drone_decision",
                        "drone_id": 0,
                        "decision": decision,
                    },
                ]
            )
            if post_rebuild is not None:
                events.append(
                    {
                        "event": "drone_post_rebuild_decision",
                        "drone_id": 0,
                        "decision": post_rebuild,
                    }
                )
            events.append({"event": "drone_action_result", "drone_id": 0})

        segment_then({"kind": "frontier", "target": target_a})
        segment_then({"kind": "frontier", "target": target_b})
        segment_then(
            {"kind": "exhausted", "target": None},
            {"kind": "step", "target": [11, 10]},
        )
        segment_then({"kind": "rotate", "target": None})

        retention = analyze_trace(events).target_retention

        self.assertEqual(retention.completed_segments, 4)
        self.assertEqual(retention.segment_followups, 4)
        self.assertEqual(retention.retained_after_segment, 1)
        self.assertEqual(retention.reranked_frontier_after_segment, 1)
        self.assertEqual(retention.local_step_after_segment, 1)
        self.assertEqual(retention.rotate_after_segment, 1)
        self.assertEqual(retention.switched_after_segment, 3)
        self.assertEqual(retention.retention_rate, 0.25)
        self.assertEqual(retention.coordinate_switch_proxy_rate, 0.75)
        self.assertEqual(retention.goal_identity_followups, 0)
        self.assertIsNone(retention.route_abandonment_rate)

    def test_stable_goal_ids_make_route_abandonment_measurable(self) -> None:
        events = [
            {
                "event": "drone_waypoint_segment_complete",
                "drone_id": 0,
                "navigation_intent": {
                    "goal_cluster_id": 7,
                    "route_edge_ids": [10, 11],
                    "edge_cursor": 0,
                    "polyline_cursor": 4,
                },
            },
            {"event": "drone_action_result", "drone_id": 0},
            {
                "event": "drone_decision",
                "drone_id": 0,
                "decision": {"kind": "step", "target": [2, 1]},
                "navigation_intent": {
                    "goal_cluster_id": 7,
                    "route_edge_ids": [10, 11],
                    "edge_cursor": 1,
                    "polyline_cursor": 0,
                },
            },
            {"event": "drone_action_result", "drone_id": 0},
            {
                "event": "drone_waypoint_segment_complete",
                "drone_id": 0,
                "navigation_intent": {
                    "goal_cluster_id": 7,
                    "route_edge_ids": [10, 11],
                    "edge_cursor": 1,
                    "polyline_cursor": 2,
                },
            },
            {"event": "drone_action_result", "drone_id": 0},
            {
                "event": "drone_decision",
                "drone_id": 0,
                "decision": {"kind": "step", "target": [3, 1]},
                "navigation_intent": {
                    "goal_cluster_id": 8,
                    "route_edge_ids": [12],
                    "edge_cursor": 0,
                    "polyline_cursor": 0,
                },
            },
            {"event": "drone_action_result", "drone_id": 0},
        ]

        retention = analyze_trace(events).target_retention

        self.assertEqual(retention.goal_identity_followups, 2)
        self.assertEqual(retention.route_abandonments, 1)
        self.assertEqual(retention.route_abandonment_rate, 0.5)
        self.assertEqual(retention.route_cursor_followups, 2)
        self.assertEqual(retention.route_cursor_continuations, 1)
        self.assertEqual(retention.route_cursor_resets_or_regressions, 1)
        self.assertEqual(retention.route_cursor_continuation_rate, 0.5)

    def test_completed_intent_clears_active_goal_before_next_selection(self) -> None:
        events = [
            {
                "event": "drone_navigation_transition",
                "drone_id": 0,
                "replan_reason": "selected",
                "intent": {"goal_cluster_id": 7},
                "mode_transition": {
                    "from_mode": None,
                    "to_mode": "travel",
                    "reason": "selected",
                },
            },
            {
                "event": "drone_navigation_transition",
                "drone_id": 0,
                "replan_reason": None,
                "intent": None,
                "navigation_intent": None,
                "mode_transition": {
                    "from_mode": "scan",
                    "to_mode": None,
                    "reason": "scan_complete",
                },
                "transition_reason": "scan_complete",
            },
            {
                "event": "drone_navigation_transition",
                "drone_id": 0,
                "replan_reason": "selected",
                "intent": {"goal_cluster_id": 8},
                "mode_transition": {
                    "from_mode": None,
                    "to_mode": "travel",
                    "reason": "selected",
                },
            },
        ]

        acceptance = analyze_trace(events).acceptance

        self.assertEqual(acceptance.goal_changes, 2)
        self.assertEqual(acceptance.unexplained_goal_changes, 0)

    def test_schema_v2_sequence_orders_concurrent_events(self) -> None:
        events = [
            {"event": "drone_action_result", "drone_id": 0, "sequence": 4},
            {
                "event": "drone_decision",
                "drone_id": 0,
                "sequence": 3,
                "decision": {"kind": "frontier", "target": [4, 5]},
            },
            {"event": "drone_action_result", "drone_id": 0, "sequence": 2},
            {
                "event": "drone_waypoint_segment_complete",
                "drone_id": 0,
                "sequence": 1,
                "target": [4, 5],
            },
        ]

        retention = analyze_trace(events).target_retention

        self.assertEqual(retention.segment_followups, 1)
        self.assertEqual(retention.retained_after_segment, 1)

    def test_trace_relative_window_end_excludes_later_runtime_events(self) -> None:
        events = [
            {
                "event": "mission_constructed",
                "mcts_decision_time_budget_ms": 25.0,
            },
            {
                "event": "sensor_scan",
                "drone_id": 0,
                "sim_time": 100.0,
                "newly_known_cells": 4,
                "confidence_gain": 1.0,
            },
            {
                "event": "drone_motion",
                "drone_id": 0,
                "sim_time": 110.0,
                "travelled_distance": 2.0,
            },
            {
                "event": "sensor_scan",
                "drone_id": 0,
                "sim_time": 120.0,
                "newly_known_cells": 100,
                "confidence_gain": 100.0,
            },
        ]

        metrics = analyze_trace(
            events,
            normalized_window_end_s=15.0,
        )

        self.assertEqual(metrics.event_count, 3)
        self.assertEqual(metrics.mcts.budget_ms, 25.0)
        self.assertEqual(metrics.information_efficiency.completed_scans, 1)
        self.assertEqual(metrics.information_efficiency.newly_known_cells, 4)
        self.assertEqual(
            metrics.information_efficiency.newly_known_cells_per_travelled_px,
            2.0,
        )

        with self.assertRaises(ValueError):
            analyze_trace(events, normalized_window_end_s=-1.0)

    def test_frontier_fallbacks_expose_repeated_and_regenerated_targets(self) -> None:
        events = [
            {
                "event": "drone_decision",
                "drone_id": 1,
                "decision": {"kind": "frontier", "target": [5, 6]},
                "mcts": {"selected_reward": 0.0},
            },
            {
                "event": "drone_frontier_reached",
                "drone_id": 1,
                "target": [5, 6],
            },
            {
                "event": "drone_decision",
                "drone_id": 1,
                "decision": {"kind": "frontier", "target": [5, 6]},
                "mcts": {"selected_reward": 0.0},
            },
            {
                "event": "drone_decision",
                "drone_id": 1,
                "decision": {"kind": "frontier", "target": [8, 9]},
                "mcts": {"selected_reward": 1.0},
            },
        ]

        fallback = analyze_trace(events).frontier_fallbacks

        self.assertEqual(fallback.fallback_count, 3)
        self.assertEqual(fallback.zero_reward_fallbacks, 2)
        self.assertEqual(fallback.unique_targets, 2)
        self.assertEqual(fallback.repeated_target_selections, 1)
        self.assertEqual(fallback.regenerated_after_reach, 1)
        self.assertEqual(len(fallback.regenerated_drone_targets), 1)
        self.assertEqual(fallback.regenerated_drone_targets[0][0], 1)

    def test_aba_window_is_trace_relative_and_resets_arrival_history(self) -> None:
        events = [
            {"event": "sensor_scan", "drone_id": 0, "sim_time": 100.0},
            {
                "event": "drone_frontier_reached",
                "drone_id": 1,
                "sim_time": 639.9,
                "target": [9, 9],
            },
            {
                "event": "drone_frontier_reached",
                "drone_id": 1,
                "sim_time": 640.0,
                "target": [1, 1],
            },
            {
                "event": "drone_frontier_reached",
                "drone_id": 1,
                "sim_time": 641.0,
                "target": [2, 2],
            },
            {
                "event": "drone_frontier_reached",
                "drone_id": 1,
                "sim_time": 642.0,
                "target": [1, 1],
            },
        ]

        reversals = analyze_trace(
            events,
            reversal_window_start_s=540.0,
        ).aba_reversals

        self.assertEqual(reversals.arrivals, 3)
        self.assertEqual(reversals.reversal_opportunities, 1)
        self.assertEqual(reversals.reversals, 1)
        self.assertEqual(reversals.reversal_rate, 1.0)
        self.assertEqual(reversals.normalized_window_start_s, 540.0)
        self.assertEqual(reversals.by_drone[0].arrivals, 0)
        self.assertIsNone(reversals.by_drone[0].reversal_rate)

    def test_mcts_metrics_use_mission_budget_and_preserve_missing_data(self) -> None:
        events = [
            {
                "event": "mission_constructed",
                "mcts_decision_time_budget_ms": 25.0,
            },
            {
                "event": "drone_decision",
                "mcts": {"elapsed_ms": 20.0, "iterations": 0},
            },
            {
                "event": "drone_decision",
                "mcts": {
                    "elapsed_ms": 30.0,
                    "iterations": 1,
                    "selected_kind": "frontier",
                    "selected_reward": 0.0,
                },
            },
            {
                "event": "drone_decision",
                "mcts": {"elapsed_ms": 50.0, "iterations": 5},
            },
            {"event": "drone_decision", "mcts": {"generated_nodes": 1}},
        ]

        mcts = analyze_trace(events).mcts

        self.assertEqual(mcts.search_count, 4)
        self.assertEqual(mcts.timing_sample_count, 3)
        self.assertEqual(mcts.missing_timing_count, 1)
        self.assertEqual(mcts.iteration_sample_count, 3)
        self.assertEqual(mcts.at_most_one_iteration_count, 2)
        self.assertAlmostEqual(mcts.at_most_one_iteration_rate, 2 / 3)
        self.assertEqual(mcts.budget_ms, 25.0)
        self.assertEqual(mcts.budget_source, "trace")
        self.assertEqual(mcts.over_budget_count, 2)
        self.assertAlmostEqual(mcts.over_budget_rate, 2 / 3)
        self.assertEqual(mcts.mean_elapsed_ms, 100 / 3)
        self.assertEqual(mcts.median_elapsed_ms, 30.0)
        self.assertEqual(mcts.p95_elapsed_ms, 50.0)
        self.assertEqual(mcts.zero_reward_frontier_fallbacks, 1)

        legacy = analyze_trace(
            [{"event": "drone_decision", "mcts": {"elapsed_ms": 41}}]
        ).mcts
        self.assertEqual(legacy.budget_ms, LEGACY_MCTS_BUDGET_MS)
        self.assertEqual(legacy.budget_source, "legacy_default")

    def test_route_metrics_measure_reuse_timing_and_segment_sources(self) -> None:
        events = [
            {
                "event": "drone_waypoint_route",
                "drone_id": 0,
                "target": [8, 8],
                "status": "ok",
                "route_elapsed_ms": 1.0,
                "cache_hit": False,
            },
            {
                "event": "drone_waypoint_route",
                "drone_id": 0,
                "target": [8, 8],
                "status": "ok",
                "route_elapsed_ms": 2.0,
                "cache_hit": True,
            },
            {
                "event": "drone_waypoint_route",
                "drone_id": 1,
                "target": [8, 8],
                "status": "disconnected",
                "route_elapsed_ms": 3.0,
            },
            {
                "event": "drone_waypoint_segment_path",
                "path_source": "astar",
                "astar_path_len": 10,
                "path_len": 10,
            },
            {
                "event": "drone_waypoint_segment_path",
                "path_source": "trusted_route_fallback",
                "astar_path_len": 0,
                "path_len": 9,
            },
        ]

        routes = analyze_trace(events).routes

        self.assertEqual(routes.route_calls, 3)
        self.assertEqual(routes.successful_routes, 2)
        self.assertEqual(routes.failed_routes, 1)
        self.assertEqual(routes.unique_drone_target_pairs, 2)
        self.assertEqual(routes.total_route_time_ms, 6.0)
        self.assertEqual(routes.mean_route_time_ms, 2.0)
        self.assertEqual(routes.median_route_time_ms, 2.0)
        self.assertEqual(routes.p95_route_time_ms, 3.0)
        self.assertEqual(routes.cache_hits, 1)
        self.assertEqual(routes.cache_misses, 1)
        self.assertEqual(routes.cache_unknown, 1)
        self.assertEqual(routes.cache_hit_rate, 0.5)
        self.assertEqual(routes.segment_calls, 2)
        self.assertEqual(routes.successful_segments, 2)
        self.assertEqual(routes.astar_attempts, 2)
        self.assertEqual(routes.astar_selected_paths, 1)
        self.assertEqual(routes.stored_polyline_selected_paths, 1)

    def test_route_acceptance_prefers_lookup_only_timing_over_repair_total(self) -> None:
        events = [
            {
                "event": "drone_waypoint_route",
                "drone_id": 0,
                "target": [8, 8],
                "status": "ok",
                "route_elapsed_ms": 80.0,
                "route_lookup_elapsed_ms": 2.0,
                "route_repair_elapsed_ms": 78.0,
                "cache_hit": False,
            },
            {
                "event": "drone_waypoint_route",
                "drone_id": 0,
                "target": [16, 8],
                "status": "ok",
                "route_elapsed_ms": 60.0,
                "route_lookup_elapsed_ms": 3.0,
                "route_repair_elapsed_ms": 57.0,
                "cache_hit": False,
            },
        ]

        metrics = analyze_trace(events)

        self.assertEqual(metrics.routes.total_route_time_ms, 5.0)
        self.assertEqual(metrics.routes.p95_route_time_ms, 3.0)
        self.assertEqual(metrics.acceptance.route_lookup_p95_ms, 3.0)

    def test_information_efficiency_requires_exact_gain_and_motion(self) -> None:
        events = [
            {
                "event": "drone_motion",
                "drone_id": 0,
                "travelled_distance": 10.0,
            },
            {
                "event": "sensor_scan",
                "drone_id": 0,
                "newly_known_cells": 8,
                "confidence_gain": 2.0,
            },
            {
                "event": "drone_motion",
                "drone_id": 0,
                "travelled_distance": 5.0,
            },
            {
                "event": "sensor_scan",
                "drone_id": 0,
                "newly_known_cells": 0,
                "confidence_gain": 0.5,
            },
        ]

        efficiency = analyze_trace(events).information_efficiency

        self.assertEqual(efficiency.travelled_distance_px, 15.0)
        self.assertEqual(efficiency.distance_source, "drone_motion")
        self.assertEqual(efficiency.completed_scans, 2)
        self.assertEqual(efficiency.gain_samples, 2)
        self.assertEqual(efficiency.scans_missing_gain, 0)
        self.assertTrue(efficiency.gain_telemetry_complete)
        self.assertEqual(efficiency.newly_known_samples, 2)
        self.assertTrue(efficiency.newly_known_telemetry_complete)
        self.assertEqual(efficiency.confidence_gain_samples, 2)
        self.assertTrue(efficiency.confidence_gain_telemetry_complete)
        self.assertEqual(efficiency.newly_known_cells, 8)
        self.assertEqual(efficiency.confidence_gain, 2.5)
        self.assertAlmostEqual(
            efficiency.newly_known_cells_per_travelled_px,
            8 / 15,
        )
        self.assertAlmostEqual(
            efficiency.confidence_gain_per_travelled_px,
            2.5 / 15,
        )

        legacy = analyze_trace(
            [
                {
                    "event": "sensor_scan",
                    "drone_id": 0,
                    "slam_updated": False,
                },
                {
                    "event": "drone_action_result",
                    "drone_id": 0,
                    "state": {"position": [10, 10]},
                },
            ]
        ).information_efficiency
        self.assertEqual(legacy.distance_source, "unavailable")
        self.assertEqual(legacy.scans_missing_gain, 1)
        self.assertFalse(legacy.gain_telemetry_complete)
        self.assertIsNone(legacy.newly_known_cells_per_travelled_px)

    def test_information_efficiency_marks_partial_signal_coverage(self) -> None:
        events = [
            {
                "event": "drone_motion",
                "drone_id": 0,
                "travelled_distance": 10.0,
            },
            {
                "event": "sensor_scan",
                "drone_id": 0,
                "newly_known_cells": 4,
                "confidence_gain": 1.0,
            },
            {
                "event": "sensor_scan",
                "drone_id": 0,
                "newly_known_cells": 2,
            },
        ]

        efficiency = analyze_trace(events).information_efficiency

        self.assertEqual(efficiency.newly_known_samples, 2)
        self.assertTrue(efficiency.newly_known_telemetry_complete)
        self.assertAlmostEqual(
            efficiency.newly_known_cells_per_travelled_px,
            0.6,
        )
        self.assertEqual(efficiency.confidence_gain_samples, 1)
        self.assertEqual(efficiency.scans_missing_confidence_gain, 1)
        self.assertFalse(efficiency.confidence_gain_telemetry_complete)
        self.assertIsNone(efficiency.confidence_gain_per_travelled_px)
        self.assertFalse(efficiency.gain_telemetry_complete)


if __name__ == "__main__":
    unittest.main()
