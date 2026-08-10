import unittest

from navigation.strategic_selection import (
    StrategicCandidate,
    select_strategic_candidates,
)


class StrategicSelectionTests(unittest.TestCase):
    def test_filters_before_bounding_and_never_scores_more_than_32(self) -> None:
        candidates = [
            StrategicCandidate(
                cluster_id=index,
                representative=(index, 0),
                expected_gain=float(100 - index),
                route_cost=float(index + 1),
                reserved_by_other=index == 0,
                reachable=index != 1,
                blacklisted=index == 2,
            )
            for index in range(50)
        ]

        selected = select_strategic_candidates(candidates, position=(49, 0))

        self.assertLessEqual(len(selected), 32)
        self.assertTrue(all(
            item.candidate.cluster_id not in {0, 1, 2}
            for item in selected
        ))

    def test_uses_locked_normalized_score_and_stable_id_ties(self) -> None:
        selected = select_strategic_candidates([
            StrategicCandidate(2, (2, 0), 10.0, 10.0),
            StrategicCandidate(1, (1, 0), 10.0, 10.0),
            StrategicCandidate(3, (3, 0), 1.0, 1.0, stall_penalty=4.0),
        ], position=(0, 0))

        self.assertEqual(
            [item.candidate.cluster_id for item in selected[:2]],
            [1, 2],
        )

    def test_wall_continuation_precedes_larger_discovery_gain(self) -> None:
        selected = select_strategic_candidates([
            StrategicCandidate(
                cluster_id=1,
                representative=(20, 0),
                expected_gain=1.0,
                route_cost=20.0,
                wall_gain=1.0,
            ),
            StrategicCandidate(
                cluster_id=2,
                representative=(1, 0),
                expected_gain=1000.0,
                route_cost=1.0,
            ),
        ], position=(0, 0))

        self.assertEqual(selected[0].candidate.cluster_id, 1)


if __name__ == "__main__":
    unittest.main()
