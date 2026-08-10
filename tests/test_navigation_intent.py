import unittest

from agents.drone_runtime_state import DroneRuntimeState
from navigation.navigation_intent import (
    MovementMode,
    MovementOutcome,
    NavigationIntent,
    NavigationWatchdog,
    TransitionReason,
)


class NavigationIntentTests(unittest.TestCase):
    def setUp(self) -> None:
        import numpy as np

        self.state = DroneRuntimeState(
            start_position=(2, 2),
            cave=np.zeros((16, 16), dtype=np.uint8),
            direction=90,
            frontier_rebuild_cooldown=0.25,
        )

    def test_intent_and_route_cursor_are_latched_in_snapshots(self) -> None:
        intent = NavigationIntent(
            mode=MovementMode.TRAVEL,
            cluster_id=17,
            gateway_id=9,
            assignment_token=31,
            target=(12, 2),
            topology_revision=4,
            requester_knowledge_revision=8,
            route_node_ids=(1, 2, 3),
            route_edge_ids=(5, 6),
            route_paths=(((2, 2), (7, 2)), ((7, 2), (12, 2))),
            route_sources=("travelled", "travelled"),
            remaining_route_cost=10.0,
            selection_slam_version=8,
            local_scan_pending=True,
            previous_primitive="deviate_left",
        )
        self.state.set_navigation_intent(intent, reason=TransitionReason.SELECTED)
        self.state.advance_navigation_intent(
            edge_cursor=1,
            polyline_cursor=0,
            remaining_route_cost=5.0,
        )

        snapshot = self.state.snapshot()

        self.assertEqual(snapshot.navigation_intent.cluster_id, 17)
        self.assertEqual(snapshot.navigation_intent.assignment_token, 31)
        self.assertEqual(snapshot.navigation_intent.edge_cursor, 1)
        self.assertEqual(snapshot.navigation_intent.route_edge_ids, (5, 6))
        self.assertEqual(
            snapshot.navigation_intent.previous_primitive,
            "deviate_left",
        )
        self.assertTrue(snapshot.navigation_intent.local_scan_pending)
        self.assertEqual(snapshot.transition_reason, TransitionReason.SELECTED)

    def test_runtime_assigns_monotonic_intent_ids_and_preserves_route_identity(
        self,
    ) -> None:
        first = NavigationIntent(
            route_id=41,
            target=(8, 2),
            route_paths=(((2, 2), (8, 2)),),
            remaining_route_cost=6.0,
        )

        latched = self.state.set_navigation_intent(
            first,
            reason=TransitionReason.SELECTED,
        )
        advanced = self.state.advance_navigation_intent(
            edge_cursor=0,
            polyline_cursor=1,
            remaining_route_cost=5.0,
        )

        self.assertGreater(latched.intent_id, 0)
        self.assertEqual(advanced.intent_id, latched.intent_id)
        self.assertEqual(advanced.route_id, 41)

        self.state.clear_navigation_intent(MovementOutcome(
            invalidated=True,
            transition_reason=TransitionReason.INVALIDATED,
        ))
        replacement = self.state.set_navigation_intent(
            NavigationIntent(route_id=42, target=(10, 2)),
            reason=TransitionReason.SELECTED,
        )

        self.assertGreater(replacement.intent_id, latched.intent_id)
        self.assertEqual(replacement.route_id, 42)

    def test_explicit_invalidation_clears_intent_but_records_outcome(self) -> None:
        self.state.set_navigation_intent(
            NavigationIntent(mode=MovementMode.HOME, target=(2, 2)),
            reason=TransitionReason.HOME,
        )
        outcome = MovementOutcome(
            invalidated=True,
            transition_reason=TransitionReason.ROUTE_EDGE_RETIRED,
        )

        previous = self.state.clear_navigation_intent(outcome)
        snapshot = self.state.snapshot()

        self.assertEqual(previous.mode, MovementMode.HOME)
        self.assertIsNone(snapshot.navigation_intent)
        self.assertEqual(snapshot.movement_mode, MovementMode.HOME)
        self.assertEqual(snapshot.last_movement_outcome, outcome)
        self.assertEqual(
            snapshot.transition_reason,
            TransitionReason.ROUTE_EDGE_RETIRED,
        )

    def test_watchdog_counts_only_gain_or_route_reduction_as_progress(self) -> None:
        watchdog = NavigationWatchdog(last_progress_time=1.0)
        watchdog = watchdog.observe(
            MovementOutcome(travelled_distance=64.0),
            now=2.0,
            visit=4,
        )
        self.assertEqual(
            watchdog.recovery_reason(now=2.0),
            TransitionReason.STALLED,
        )

        watchdog = watchdog.observe(
            MovementOutcome(route_progress_delta=1.0),
            now=3.0,
            visit=5,
        )
        self.assertEqual(watchdog.distance_without_progress, 0.0)
        self.assertIsNone(watchdog.recovery_reason(now=3.0))

    def test_second_a_b_a_reversal_requests_recovery(self) -> None:
        watchdog = NavigationWatchdog(last_progress_time=1.0)
        for visit in (1, 2, 1, 2, 1):
            watchdog = watchdog.observe(
                MovementOutcome(route_progress_delta=1.0),
                now=2.0,
                visit=visit,
            )

        self.assertEqual(watchdog.reversal_count, 3)
        self.assertEqual(
            watchdog.recovery_reason(now=2.0),
            TransitionReason.REVERSAL,
        )
        self.assertGreaterEqual(watchdog.revisit_ratio, 0.0)


if __name__ == "__main__":
    unittest.main()
