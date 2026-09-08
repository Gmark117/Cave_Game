import unittest

import numpy as np

from mapping.frontiers import significant_frontier_mask
from mapping.slam_map import FREE, OCCUPIED, UNKNOWN


class SignificantFrontierTests(unittest.TestCase):
    def test_discards_tiny_component_with_tiny_unknown_support(self) -> None:
        occupancy = np.full((32, 32), FREE, dtype=np.int8)
        confidence = np.ones((32, 32), dtype=np.float32)
        occupancy[16, 16] = UNKNOWN
        confidence[16, 16] = 0.0

        frontier, diagnostics = significant_frontier_mask(
            occupancy,
            confidence,
            0.6,
            minimum_component_cells=12,
            minimum_unknown_support_cells=64,
        )

        self.assertFalse(frontier.any())
        self.assertEqual(diagnostics.raw_frontier_pixels, 8)
        self.assertEqual(diagnostics.component_sizes, (8,))
        self.assertEqual(diagnostics.discarded_component_count, 1)
        self.assertEqual(diagnostics.discarded_frontier_pixels, 8)

    def test_preserves_tiny_doorway_into_large_unknown_region(self) -> None:
        occupancy = np.full((32, 32), OCCUPIED, dtype=np.int8)
        confidence = np.ones((32, 32), dtype=np.float32)
        occupancy[15, 15] = FREE
        occupancy[16:26, 16:26] = UNKNOWN
        confidence[16:26, 16:26] = 0.0

        frontier, diagnostics = significant_frontier_mask(
            occupancy,
            confidence,
            0.6,
            minimum_component_cells=12,
            minimum_unknown_support_cells=64,
        )

        self.assertTrue(frontier[15, 15])
        self.assertEqual(diagnostics.raw_frontier_pixels, 1)
        self.assertEqual(diagnostics.significant_frontier_pixels, 1)
        self.assertEqual(
            diagnostics.unknown_supported_component_count,
            1,
        )
        self.assertEqual(
            diagnostics.unknown_supported_candidate_count,
            1,
        )
        self.assertEqual(diagnostics.frontier_unknown_basin_count, 1)
        self.assertEqual(diagnostics.significant_unknown_basin_count, 1)
        self.assertEqual(diagnostics.discarded_component_count, 0)

    def test_keeps_only_one_small_gateway_per_unknown_basin(self) -> None:
        occupancy = np.full((40, 40), OCCUPIED, dtype=np.int8)
        confidence = np.ones((40, 40), dtype=np.float32)
        occupancy[11:31, 8:32] = UNKNOWN
        confidence[11:31, 8:32] = 0.0
        occupancy[10, 10] = FREE
        occupancy[10, 28] = FREE

        frontier, diagnostics = significant_frontier_mask(
            occupancy,
            confidence,
            0.6,
            minimum_component_cells=12,
            minimum_unknown_support_cells=64,
        )

        self.assertTrue(frontier[10, 10])
        self.assertFalse(frontier[10, 28])
        self.assertEqual(diagnostics.raw_component_count, 2)
        self.assertEqual(diagnostics.unknown_supported_candidate_count, 2)
        self.assertEqual(diagnostics.unknown_supported_component_count, 1)
        self.assertEqual(
            diagnostics.redundant_unknown_supported_component_count,
            1,
        )
        self.assertEqual(diagnostics.frontier_unknown_basin_count, 1)
        self.assertEqual(diagnostics.significant_unknown_basin_count, 1)
        self.assertEqual(
            [item.retention_reason for item in diagnostics.components],
            [
                "unknown_basin_gateway",
                "duplicate_unknown_basin_gateway",
            ],
        )
        self.assertEqual(
            diagnostics.components[0].bounding_box,
            (10, 10, 10, 10),
        )

    def test_large_frontier_already_represents_its_unknown_basin(self) -> None:
        occupancy = np.full((40, 40), OCCUPIED, dtype=np.int8)
        confidence = np.ones((40, 40), dtype=np.float32)
        occupancy[11:31, 5:35] = UNKNOWN
        confidence[11:31, 5:35] = 0.0
        occupancy[10, 5:17] = FREE
        occupancy[10, 30] = FREE

        frontier, diagnostics = significant_frontier_mask(
            occupancy,
            confidence,
            0.6,
            minimum_component_cells=12,
            minimum_unknown_support_cells=64,
        )

        self.assertEqual(int(np.count_nonzero(frontier)), 12)
        self.assertEqual(diagnostics.large_component_count, 1)
        self.assertEqual(diagnostics.unknown_supported_candidate_count, 1)
        self.assertEqual(diagnostics.unknown_supported_component_count, 0)
        self.assertEqual(
            diagnostics.redundant_unknown_supported_component_count,
            1,
        )

    def test_border_connected_unknown_does_not_rescue_small_gateway(
        self,
    ) -> None:
        occupancy = np.full((32, 32), OCCUPIED, dtype=np.int8)
        confidence = np.ones((32, 32), dtype=np.float32)
        occupancy[16:32, 16:26] = UNKNOWN
        confidence[16:32, 16:26] = 0.0
        occupancy[15, 16] = FREE

        frontier, diagnostics = significant_frontier_mask(
            occupancy,
            confidence,
            0.6,
            minimum_component_cells=12,
            minimum_unknown_support_cells=64,
        )

        self.assertFalse(frontier.any())
        self.assertEqual(diagnostics.raw_component_count, 1)
        self.assertEqual(
            diagnostics.border_connected_unknown_basin_count,
            1,
        )
        self.assertEqual(
            diagnostics.border_connected_unknown_supported_candidate_count,
            1,
        )
        component = diagnostics.components[0]
        self.assertTrue(component.touches_border_connected_unknown)
        self.assertEqual(
            component.retention_reason,
            "border_connected_unknown_basin",
        )
        self.assertEqual(
            component.maximum_rescuable_unknown_support_cells,
            0,
        )

    def test_large_component_survives_border_connected_unknown(self) -> None:
        occupancy = np.full((32, 32), OCCUPIED, dtype=np.int8)
        confidence = np.ones((32, 32), dtype=np.float32)
        occupancy[16:32, 8:24] = UNKNOWN
        confidence[16:32, 8:24] = 0.0
        occupancy[15, 8:20] = FREE

        frontier, diagnostics = significant_frontier_mask(
            occupancy,
            confidence,
            0.6,
            minimum_component_cells=12,
            minimum_unknown_support_cells=64,
        )

        self.assertEqual(int(np.count_nonzero(frontier)), 12)
        self.assertEqual(diagnostics.large_component_count, 1)
        self.assertEqual(
            diagnostics.components[0].retention_reason,
            "large_component",
        )


if __name__ == "__main__":
    unittest.main()
