"""Synthetic replay tests for structured runtime-trace characterization."""

from __future__ import annotations

import math
import unittest

from tools.analyze_runtime_trace import (
    LEGACY_MCTS_BUDGET_MS,
    analyze_trace,
    format_characterization,
    summarize,
)


class RuntimeTraceAnalysisTests(unittest.TestCase):
    def test_summary_reports_focused_endgame_economics(self) -> None:
        report = "\n".join(summarize([
            {
                "event": "rover_focused_endgame_changed",
                "active": True,
            },
            {
                "event": "rover_focused_endgame_assignment",
                "estimated_outbound_cost": 12.5,
                "estimated_round_trip_cost": 25.0,
            },
            {"event": "rover_focused_endgame_staging_selected"},
            {"event": "rover_focused_endgame_staging_held"},
            {"event": "rover_route_negative_cached"},
            {"event": "rover_route_negative_cache_hit"},
            {"event": "rover_frontier_route_retry_suppressed"},
            {
                "event": "rover_task_completed",
                "outbound_distance": 12.0,
                "service_distance": 7.0,
                "return_distance": 15.0,
            },
            {
                "event": "rover_focused_endgame_changed",
                "active": False,
            },
        ]))

        self.assertIn(
            "focused endgame economics: activations=1 deactivations=1 "
            "assignments=1 estimated_outbound=12.5px "
            "estimated_round_trip=25.0px staging_selected=1 "
            "staging_held=1 route_cached=1 cache_hits=1 "
            "retry_suppressed=1",
            report,
        )
        self.assertIn(
            "reported sortie distance: outbound=12.0px service=7.0px "
            "return=15.0px",
            report,
        )

    def test_summary_reports_component_rounds_lineage_and_quiescence(self) -> None:
        report = "\n".join(summarize([
            {
                "event": "rover_discovery_round_started",
                "round_id": 1,
                "round_kind": "rover_scan",
            },
            {
                "event": "rover_discovery_round_completed",
                "round_id": 1,
                "round_kind": "rover_scan",
            },
            {
                "event": "rover_frontier_registry_reconciled",
                "revision": 3,
                "active_component_count": 2,
                "ready_work_unit_count": 4,
                "dormant_component_count": 1,
                "resolved_component_count": 5,
                "exploration_modes": {
                    "focused": 1,
                    "wall_follow": 0,
                    "sweep": 1,
                },
            },
            {
                "event": "rover_frontier_lineage_changed",
                "transition_kind": "split",
            },
            {
                "event": "rover_task_claimed",
                "drone_id": 0,
                "claim_token": 10,
            },
            {
                "event": "rover_task_completed",
                "drone_id": 0,
                "suspended": True,
                "suspension_reason": "energy_reserve",
                "work_unit_outcomes": [{"disposition": "zero_gain"}],
            },
            {
                "event": "drone_component_directive_started",
                "drone_id": 0,
                "waited_seconds": 2.5,
            },
            {
                "event": "drone_component_target_adjusted_after_share",
                "drone_id": 0,
            },
            {
                "event": "drone_component_target_retired_after_share",
                "drone_id": 1,
            },
            {
                "event": "drone_component_sweep_advanced",
                "drone_id": 0,
            },
            {
                "event": "drone_dfs_backtrack_replanned",
                "drone_id": 0,
                "recovered": True,
            },
            {
                "event": "drone_dfs_popped",
                "drone_id": 0,
            },
            {
                "event": "drone_dfs_reposition_started",
                "drone_id": 0,
            },
            {
                "event": "drone_motion",
                "drone_id": 0,
                "source": "component_dfs_reposition_astar",
                "travelled_distance": 12.5,
            },
            {
                "event": "drone_component_check_in_queued",
                "drone_id": 0,
            },
            {
                "event": "rover_component_check_in_processed",
                "drone_id": 0,
                "queue_wait_ms": 3.0,
                "processing_ms": 7.0,
            },
            {
                "event": "rover_frontier_target_acquired",
                "rover_id": 0,
            },
            {
                "event": "rover_frontier_staging_reached",
                "rover_id": 0,
            },
            {
                "event": "rover_exploration_quiescence_evaluated",
                "reason": "wall_goal_and_no_tasks",
                "ready_task_count": 0,
                "live_claim_count": 0,
            },
        ]))

        self.assertIn("Component exploration:", report)
        self.assertIn(
            "discovery rounds: started=1 completed=1 rover_scan=1 radial_probe=0",
            report,
        )
        self.assertIn("revision=3 active_components=2 ready_units=4", report)
        self.assertIn("outcomes={'zero_gain': 1}", report)
        self.assertIn("suspensions={'energy_reserve': 1}", report)
        self.assertIn("lineage: {'split': 1}", report)
        self.assertIn(
            "local autonomy: shared_adjustments=1 shared_retirements=1 "
            "sweep_advances=1 backtrack_replans=1",
            report,
        )
        self.assertIn(
            "DFS navigation: logical_pops=1 repositions=1 "
            "astar_px=12.5 breadcrumb_px=0.0 legacy_retrace_px=0.0",
            report,
        )
        self.assertIn(
            "async rover check-ins: queued=1 processed=1 "
            "queue_wait_avg=3.00ms queue_wait_max=3.00ms "
            "processing_avg=7.00ms processing_max=7.00ms",
            report,
        )
        self.assertIn(
            (
                "moving rovers: targets=1 departures_authorized=0 "
                "reached=1 invalidated=0 routes=0"
            ),
            report,
        )
        self.assertIn("completed wait by drone: 0=2.50s", report)
        self.assertIn("max_ready_tasks=0 max_live_claims=0", report)

    def test_summary_reports_compact_docking_sessions(self) -> None:
        report = "\n".join(summarize([
            {
                "event": "drone_docked",
                "drone_id": 0,
                "source": "check_in_intercept",
            },
            {
                "event": "drone_docked",
                "drone_id": 1,
                "source": "report_endpoint",
            },
            {
                "event": "drone_undocked",
                "drone_id": 0,
                "reason": "directive_assigned",
                "carried_distance": 12.5,
                "carried_steps": 4,
                "docked_seconds": 8.0,
                "learned_endpoint_epochs": [2, 3],
            },
        ]))

        self.assertIn(
            "docking: acquired=2 released=1 active_at_trace_end=1 "
            "intercepts=1 carried=12.5px steps=4 docked=8.00s",
            report,
        )
        self.assertIn("learned_epochs=[2, 3]", report)
        self.assertIn("'check_in_intercept': 1", report)
        self.assertIn("releases={'directive_assigned': 1}", report)

    def test_summary_reports_mid_route_peer_share_interruptions(self) -> None:
        report = "\n".join(summarize([
            {
                "event": "drone_route_interrupted_after_share",
                "drone_id": 0,
                "source": "component_task_transit",
            },
            {
                "event": "drone_route_interrupted_after_share",
                "drone_id": 2,
                "source": "component_dfs_reposition_astar",
            },
        ]))

        self.assertIn(
            "mid-route peer sharing: interruptions=2 sources={"
            "'component_dfs_reposition_astar': 1, "
            "'component_task_transit': 1}",
            report,
        )

    def test_summary_reports_focused_observation_route_savings(self) -> None:
        report = "\n".join(summarize([
            {
                "event": "drone_component_observation_pose_selected",
                "source": "component_task_transit",
                "saved_route_distance": 12.5,
            },
            {
                "event": "drone_component_observation_pose_selected",
                "source": "component_dfs_reposition_astar",
                "saved_route_distance": 0.0,
            },
        ]))

        self.assertIn(
            "focused observation poses: selected=2 shortened=1 "
            "route_suffix_avoided=12.5px sources={"
            "'component_dfs_reposition_astar': 1, "
            "'component_task_transit': 1}",
            report,
        )

    def test_epoch_publication_reports_standby_and_open_wait_at_completion(self) -> None:
        report = "\n".join(summarize([
            {"event": "rover_sector_epoch_published", "generation": 2, "sim_time": 10.0,
             "assignments": [
                 {"drone_id": 0, "frontier_cells": 1, "standby": False,
                  "scope_pixels": 100, "estimated_effort": 8.0},
                 {"drone_id": 1, "frontier_cells": 0, "standby": True},
             ]},
            {"event": "drone_sector_standby", "drone_id": 1, "generation": 2, "sim_time": 10.0},
            {"event": "drone_sector_assigned", "drone_id": 0, "generation": 2,
             "frontier_cells": 1, "sim_time": 12.0},
            {"event": "drone_frontiers_rebuilt", "drone_id": 0, "sim_time": 13.0,
             "scope_excluded_frontier_count": 7},
            {"event": "drone_motion", "drone_id": 0, "sim_time": 14.0, "travelled_distance": 20.0},
            {"event": "drone_sector_waiting_for_team", "drone_id": 0,
             "generation": 2, "sim_time": 15.0},
            {"event": "team_wall_mapping_tolerance_reached", "sim_time": 20.0},
            {"event": "trace_closed", "sim_time": 22.0},
        ]))
        self.assertIn("generation 2: duration=10.00s", report)
        self.assertIn("assigned_frontiers=1", report)
        self.assertIn("active=[0] standby=[1]", report)
        self.assertIn("scope_pixels=100 estimated_effort=8.00", report)
        self.assertIn("excluded_frontier_samples=7", report)
        self.assertIn("barrier_wait=5.00s standby_time=10.00s", report)
        self.assertIn("standby_distance=0.00px", report)
        self.assertIn("sector standby: count=1 total=10.00s", report)

    def test_summary_reports_arrival_departure_sharing_protocol(self) -> None:
        report = "\n".join(summarize([
            {
                "event": "drone_rover_check_in",
                "drone_id": 0,
                "reason": "exchanged",
            },
            {
                "event": "drone_rover_departure_share",
                "drone_id": 0,
                "reason": "unchanged_versions",
            },
            {
                "event": "drone_sharing_suppressed",
                "drone_id": 0,
                "reason": "rover_arrival_or_standby",
            },
        ]))

        self.assertIn("Sharing protocol:", report)
        self.assertIn("rover arrivals: total=1 exchanged=1", report)
        self.assertIn(
            "rover departures: total=1 unchanged_versions=1",
            report,
        )
        self.assertIn("rover-area pair suppressions: total=1", report)

    def test_summary_reports_rover_frontier_significance(self) -> None:
        report = "\n".join(summarize([
            {
                "event": "rover_sector_frontiers_filtered",
                "generation": 1,
                "raw_frontier_pixels": 11,
                "raw_component_count": 2,
                "significant_frontier_pixels": 0,
                "significant_component_count": 0,
                "unknown_supported_component_count": 0,
                "unknown_supported_candidate_count": 7,
                "redundant_unknown_supported_component_count": 7,
                "frontier_unknown_basin_count": 2,
                "significant_unknown_basin_count": 0,
                "border_connected_unknown_basin_count": 1,
                "border_connected_unknown_supported_candidate_count": 3,
                "discarded_frontier_pixels": 11,
                "discarded_component_count": 2,
                "minimum_component_cells": 12,
                "minimum_unknown_support_cells": 64,
                "mission_exhausted": True,
            },
            {
                "event": "rover_sector_workload_balanced",
                "generation": 1,
                "initial_frontier_workloads": [30, 0, 0],
                "balanced_frontier_workloads": [10, 10, 10],
                "initial_workload_spread": 30,
                "balanced_workload_spread": 0,
                "initial_estimated_efforts": [45.0, 5.0, 5.0],
                "balanced_estimated_efforts": [20.0, 18.0, 17.0],
                "initial_estimated_effort_spread": 40.0,
                "balanced_estimated_effort_spread": 3.0,
                "moved_coarse_cell_count": 2,
            },
            {
                "event": "rover_sector_frontier_outcomes",
                "generation": 2,
                "previous_generation": 1,
                "confident_occupied_gain": 0,
                "remembered_component_count": 2,
                "suppressed_component_count": 2,
                "suppressed_frontier_pixels": 11,
                "remaining_component_count": 0,
                "remaining_frontier_pixels": 0,
                "evaluated_component_count": 2,
                "locally_unchanged_component_count": 1,
                "reported_zero_gain_component_count": 1,
                "productive_component_count": 0,
                "resolved_component_count": 0,
                "component_outcomes": [{
                    "component_id": 7,
                    "previous_size": 6,
                    "current_size": 6,
                    "overlap_iou": 1.0,
                    "local_confident_cell_gain": 0,
                    "local_confident_occupied_gain": 0,
                    "local_confidence_gain": 0.0,
                    "suppression_reasons": ["zero_gain_directed_scan"],
                    "disposition": "reported_zero_gain",
                }],
                "mission_exhausted": True,
            },
        ]))

        self.assertIn("Rover frontier significance:", report)
        self.assertIn("generation 1: raw=11px/2 components", report)
        self.assertIn("discarded=11px/2 components", report)
        self.assertIn("rescue_candidates=7", report)
        self.assertIn("duplicate_gateways=7", report)
        self.assertIn("unknown_basins=0/2", report)
        self.assertIn("border_unknown_basins=1", report)
        self.assertIn("border_rescues_rejected=3", report)
        self.assertIn("thresholds=12px-or-64unknown", report)
        self.assertIn("mission_exhausted=True", report)
        self.assertIn("Rover sector workload balance:", report)
        self.assertIn("spread=30->0", report)
        self.assertIn(
            "estimated_effort=[45.0, 5.0, 5.0]->[20.0, 18.0, 17.0]",
            report,
        )
        self.assertIn("effort_spread=40.00->3.00", report)
        self.assertIn("moved_cells=2", report)
        self.assertIn("Rover frontier outcome memory:", report)
        self.assertIn("occupied_gain=0", report)
        self.assertIn("suppressed=11px/2 components", report)
        self.assertIn("unchanged=1", report)
        self.assertIn("reported_zero_gain=1", report)
        self.assertIn("component 7: reported_zero_gain", report)

    def test_summary_reports_frontier_arrivals_and_route_circuity(
        self,
    ) -> None:
        report = "\n".join(summarize([
            {
                "event": "drone_border_path",
                "drone_id": 1,
                "path_status": "complete",
                "route_circuity": 7.99,
            },
            {
                "event": "drone_frontier_route_rejected",
                "drone_id": 1,
            },
            {
                "event": "drone_frontier_reached",
                "drone_id": 1,
                "target": [10, 20],
            },
        ]))

        self.assertIn("drone_frontier_reached: 1", report)
        self.assertIn("drone_frontier_route_rejected: 1", report)
        self.assertIn("frontier route circuity:", report)
        self.assertIn("max=7.99x", report)
        self.assertIn("over_4x=1", report)

    def test_summary_reports_sector_epoch_yield_and_wait_time(self) -> None:
        report = "\n".join(summarize([
            {
                "event": "drone_sector_assigned",
                "drone_id": 0,
                "generation": 1,
                "frontier_cells": 12,
                "sim_time": 10.0,
            },
            {
                "event": "drone_motion",
                "drone_id": 0,
                "travelled_distance": 20.0,
                "sim_time": 11.0,
            },
            {
                "event": "sensor_scan",
                "drone_id": 0,
                "newly_known_cells": 5,
                "confidence_gain": 5.0,
                "sim_time": 12.0,
            },
            {
                "event": "drone_sector_wait_completed",
                "drone_id": 0,
                "waited_seconds": 4.5,
                "sim_time": 13.0,
            },
            {
                "event": "mission_shutdown_started",
                "sim_time": 14.0,
            },
        ]))

        self.assertIn("Sector epoch yield:", report)
        self.assertIn("generation 1: duration=4.00s", report)
        self.assertIn("assigned_frontiers=12", report)
        self.assertIn("sensor_gain=5", report)
        self.assertIn("gain_per_px=0.2500", report)
        self.assertIn("scope_pixels=N/A", report)
        self.assertIn("effort_terms=[N/A]", report)
        self.assertIn("sector barrier waits:", report)
        self.assertIn("total=4.50s", report)

    def test_unfinished_epoch_uses_last_observation_as_end(self) -> None:
        report = "\n".join(summarize([
            {"event": "drone_sector_assigned", "drone_id": 0,
             "generation": 1, "frontier_cells": 12, "sim_time": 10.0},
            {"event": "sensor_scan", "drone_id": 0, "sim_time": 25.0},
        ]))
        self.assertIn("generation 1: duration=15.00s", report)

    def test_endgame_cost_uses_offline_floor_and_rover_local_gain(self) -> None:
        report = "\n".join(summarize([
            {"event": "mission_run_started", "sim_time": 0.0},
            {"event": "frame_summary", "sim_time": 10.0,
             "floor_exploration_ratio": 0.97,
             "rover_states": [{"position": [0, 0]}]},
            {"event": "drone_component_check_in", "sim_time": 19.0,
             "rover_slam_newly_known_cells": 100,
             "rover_terrain_known_floor_cells": 1000},
            {"event": "frame_summary", "sim_time": 20.0,
             "floor_exploration_ratio": 0.98,
             "rover_states": [{"position": [0, 0]}]},
            {"event": "drone_motion", "sim_time": 25.0,
             "started_sim_time": 15.0, "ended_sim_time": 25.0,
             "travelled_distance": 100.0},
            {"event": "sensor_scan", "sim_time": 25.0,
             "newly_known_cells": 5},
            {"event": "rover_task_claimed", "sim_time": 25.0},
            {"event": "frame_summary", "sim_time": 30.0,
             "floor_exploration_ratio": 0.985,
             "rover_states": [{"position": [3, 4]}]},
            {"event": "drone_motion", "sim_time": 40.0,
             "started_sim_time": 30.0, "ended_sim_time": 40.0,
             "travelled_distance": 40.0},
            {"event": "sensor_scan", "sim_time": 35.0,
             "newly_known_cells": 3},
            {"event": "rover_task_claimed", "sim_time": 36.0},
            {"event": "rover_task_completed", "sim_time": 37.0},
            {"event": "rover_rendezvous_endpoint_proposed",
             "sim_time": 38.0},
            {"event": "drone_component_check_in", "sim_time": 39.0,
             "rover_slam_newly_known_cells": 108,
             "rover_terrain_known_floor_cells": 1020},
            {"event": "frame_summary", "sim_time": 40.0,
             "floor_exploration_ratio": 0.989,
             "rover_states": [{"position": [6, 8]}]},
            {"event": "exploration_complete_presented", "sim_time": 40.5,
             "floor_exploration_ratio": 0.989},
        ]))

        self.assertIn("Endgame cost (offline floor", report)
        self.assertIn(
            "from 98.00% at t=20.0s: floor_gain=0.900pp "
            "elapsed=20.5s drone=90px rover_sampled=10px "
            "sensor_slam_gain=8 rover_known_slam_delta=8 "
            "rover_known_terrain_delta=20 claims=2 reports=1 proposals=1",
            report,
        )
        self.assertIn("from 98.50% at t=30.0s:", report)

    def test_rendezvous_summary_flags_fallback_during_queued_check_in(
        self,
    ) -> None:
        report = "\n".join(summarize([
            {"event": "drone_component_check_in_queued", "drone_id": 0,
             "sim_time": 10.0},
            {"event": "drone_rendezvous_endpoint_fallback", "drone_id": 0,
             "sim_time": 10.005, "target_source": "rover_confirmed_relay"},
            {"event": "drone_rendezvous_endpoint_fallback", "drone_id": 1,
             "sim_time": 11.0, "target_source": "proposal_after_absence"},
        ]))

        self.assertIn(
            "rendezvous fallbacks: sources={'proposal_after_absence': 1, "
            "'rover_confirmed_relay': 1} same_tick_queued_checkins=1",
            report,
        )

    def test_summary_reports_coverage_memory_and_ingress_recovery(self) -> None:
        report = "\n".join(summarize([
            {
                "event": "drone_random_direction_selected",
                "drone_id": 0,
                "selection_mode": "wall_tracking",
                "selected_coverage_visit_pressure": 2.0,
                "selected_coverage_edge_pressure": 1.0,
                "selected_coverage_penalty_factor": 0.25,
            },
            {
                "event": "drone_motion",
                "drone_id": 0,
                "travelled_distance": 40.0,
                "coverage_cell_entries": 4,
                "coverage_new_cell_entries": 1,
                "coverage_revisit_entries": 3,
                "coverage_repeated_edge_entries": 2,
                "coverage_known_cell_count": 8,
                "coverage_known_edge_count": 7,
            },
            {
                "event": "drone_sector_ingress_recovery",
                "drone_id": 0,
                "path_status": "complete",
            },
        ]))

        self.assertIn("drone_sector_ingress_recovery: 1", report)
        self.assertIn("coverage memory:", report)
        self.assertIn("penalized=1", report)
        self.assertIn("avg_factor=0.250", report)
        self.assertIn("revisits=3/4 (75.0%)", report)
        self.assertIn("repeated_edges=2/4 (50.0%)", report)
        self.assertIn("sector_ingress:complete=1", report)

    def test_summary_reports_pre_scan_heading_restoration(self) -> None:
        report = "\n".join(summarize([
            {
                "event": "drone_stagnation_scan_exit_reoriented",
                "drone_id": 1,
                "exact_resume": True,
                "resume_heading_delta": 0.0,
            },
            {
                "event": "drone_stagnation_scan_exit_reoriented",
                "drone_id": 1,
                "exact_resume": False,
                "resume_heading_delta": 30.0,
            },
            {
                "event": "drone_stagnation_scan_timed_out",
                "drone_id": 1,
            },
        ]))

        self.assertIn("drone_stagnation_scan_timed_out: 1", report)
        self.assertIn("directed scan exits:", report)
        self.assertIn("exact_resume=1/2", report)
        self.assertIn("fallback=1", report)
        self.assertIn("avg_delta=15.0deg", report)
        self.assertIn("max_delta=30.0deg", report)

    def test_summary_reports_sensor_timings_and_tolerated_completion(
        self,
    ) -> None:
        report = "\n".join(summarize([
            {
                "event": "sensor_scan",
                "drone_id": 0,
                "vision_elapsed_ms": 10.0,
                "slam_elapsed_ms": 4.0,
                "terrain_elapsed_ms": 6.0,
                "sensor_elapsed_ms": 20.0,
            },
            {
                "event": "team_wall_mapping_tolerance_reached",
                "mapped_wall_pixels": 9414,
                "total_wall_pixels": 9444,
                "missing_wall_pixels": 30,
            },
        ]))

        self.assertIn("Completion trigger:", report)
        self.assertIn("missing=30", report)
        self.assertIn("sensor scan timings:", report)
        self.assertIn("avg_total=20.00ms", report)

    def test_component_policy_treats_wall_events_as_diagnostics(self) -> None:
        report = "\n".join(summarize([
            {
                "event": "mission_constructed",
                "sim_time": 0.0,
                "exploration_completion": (
                    "physical_team_quiescence_after_component_exhaustion"
                ),
            },
            {"event": "mission_run_started", "sim_time": 0.0},
            {
                "event": "team_wall_mapping_complete",
                "sim_time": 10.0,
                "mapped_wall_pixels": 100,
                "total_wall_pixels": 100,
            },
            {
                "event": "exploration_complete_presented",
                "sim_time": 20.0,
                "floor_exploration_ratio": 0.987,
            },
        ]))

        self.assertIn("completion=20.00s", report)
        self.assertIn(
            "Completion trigger: exploration_complete_presented floor=98.70%",
            report,
        )
        self.assertNotIn("mapped=100/100", report)

        stopped_early = "\n".join(summarize([
            {
                "event": "mission_constructed",
                "sim_time": 0.0,
                "exploration_completion": (
                    "physical_team_quiescence_after_component_exhaustion"
                ),
            },
            {"event": "mission_run_started", "sim_time": 0.0},
            {
                "event": "team_wall_mapping_complete",
                "sim_time": 10.0,
                "mapped_wall_pixels": 100,
                "total_wall_pixels": 100,
            },
            {"event": "mission_shutdown_complete", "sim_time": 12.0},
        ]))
        self.assertNotIn("completion=", stopped_early)
        self.assertNotIn("Completion trigger:", stopped_early)

    def test_summary_reports_global_commitment_and_endgame_routes(self) -> None:
        report = "\n".join(summarize([
            {
                "event": "drone_global_frontiers_rebuilt",
                "drone_id": 0,
                "elapsed_ms": 10.0,
                "region_count": 4,
                "eligible_region_count": 2,
                "filtered_region_count": 2,
                "selected_target_retained": True,
                "forced": True,
                "suppressed_region_count": 1,
            },
            {
                "event": "drone_local_frontier_exhaustion_deferred",
                "drone_id": 0,
            },
            {
                "event": "drone_global_frontier_path",
                "drone_id": 0,
                "path_status": "complete",
            },
        ]))

        self.assertIn("drone_local_frontier_exhaustion_deferred: 1", report)
        self.assertIn("drone_global_frontier_path: 1", report)
        self.assertIn("retained=1/1", report)
        self.assertIn("forced=1", report)
        self.assertIn("avg_suppressed=1.0", report)
        self.assertIn("global:complete=1", report)

    def test_summary_reports_active_global_target_ownership(self) -> None:
        report = "\n".join(summarize([{
            "event": "drone_random_direction_selected",
            "drone_id": 0,
            "global_frontier_active": True,
            "global_frontier_region_size": 100,
            "global_frontier_region_distance": 80.0,
            "global_frontier_ownership_margin": 0.25,
            "global_frontier_launch_sector_alignment": 0.75,
            "global_frontier_ownership_contribution": 0.5,
            "global_frontier_requester_distance": 80.0,
            "global_frontier_nearest_peer_distance": 100.0,
        }]))

        self.assertIn("active global target ownership:", report)
        self.assertIn("owned=100.0%", report)
        self.assertIn("avg_margin=0.250", report)
        self.assertIn("avg_requester_distance=80.0px", report)
        self.assertIn("avg_nearest_peer_distance=100.0px", report)

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

    def test_trace_analysis_uses_the_complete_runtime(self) -> None:
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

        metrics = analyze_trace(events)

        self.assertEqual(metrics.event_count, 4)
        self.assertEqual(metrics.mcts.budget_ms, 25.0)
        self.assertEqual(metrics.information_efficiency.completed_scans, 2)
        self.assertEqual(metrics.information_efficiency.newly_known_cells, 104)
        self.assertEqual(
            metrics.information_efficiency.newly_known_cells_per_travelled_px,
            52.0,
        )

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

    def test_summary_aggregates_incidental_scan_outcomes(self) -> None:
        lines = summarize([
            {
                "event": "drone_incidental_scan_candidate",
                "drone_id": 0,
                "pocket_cells": 9,
            },
            {
                "event": "drone_incidental_scan_finished",
                "drone_id": 0,
                "outcome": "completed",
                "wait_seconds": 1.25,
                "requested_rotation": 45.0,
                "newly_known_cells": 7,
                "original_pocket_cells_closed": 5,
            },
            {
                "event": "drone_incidental_route_resumed",
                "drone_id": 0,
                "disposition": "retained",
            },
        ])

        summary = "\n".join(lines)
        self.assertIn("incidental_scan_summary", summary)
        self.assertIn("completed=1", summary)
        self.assertIn("wait=1.250s", summary)
        self.assertIn("predicted_support=9", summary)
        self.assertIn("pocket_cells_closed=5", summary)
        self.assertIn("retained=1", summary)


if __name__ == "__main__":
    unittest.main()
