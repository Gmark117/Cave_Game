import unittest

import numpy as np

from mapping.frontier_registry import (
    CausalTransition,
    ComponentState,
    ExplorationMode,
    FrontierComponentRegistry,
    LineageKind,
    SensorFootprint,
    WorkUnitState,
)
from mapping.slam_map import FREE, OCCUPIED, UNKNOWN, SlamSnapshot


def line_slam(
    shape: tuple[int, int],
    cells: tuple[tuple[int, int], ...],
    *,
    version: int,
    occupied: tuple[tuple[int, int], ...] = (),
) -> SlamSnapshot:
    occupancy = np.full(shape, UNKNOWN, dtype=np.int8)
    confidence = np.zeros(shape, dtype=np.float32)
    for x, y in cells:
        occupancy[y, x] = FREE
        confidence[y, x] = 1.0
    for x, y in occupied:
        occupancy[y, x] = OCCUPIED
        confidence[y, x] = 1.0
    return SlamSnapshot(occupancy, confidence, version=version)


class FrontierComponentRegistryTests(unittest.TestCase):
    def registry(
        self,
        *,
        shape: tuple[int, int] = (32, 32),
        sensor_range: float = 5.0,
    ) -> FrontierComponentRegistry:
        return FrontierComponentRegistry(
            shape,
            SensorFootprint(
                max_range=sensor_range,
                fov_deg=60.0,
                frontier_stride=1,
                global_cell_size=8,
            ),
            minimum_component_cells=1,
            minimum_unknown_support_cells=1,
        )

    def _two_low_gain_lineage_scans(
        self,
        scan_slam: SlamSnapshot,
    ) -> tuple[FrontierComponentRegistry, int]:
        registry = self.registry(sensor_range=20.0)
        cells = tuple((x, 8) for x in range(5, 13))
        initial = registry.reconcile(line_slam((32, 32), cells, version=1))
        parent_id = initial.active_component_ids[0]
        geometry = registry.components[parent_id].geometry
        unknown = registry._unknown_mask(scan_slam)
        for _index in range(2):
            unit = registry.work_units[
                registry.components[parent_id].work_unit_ids[0]
            ]
            registry.record_scan_result(
                unit.work_unit_id,
                scan_position=(1, 1),
                frontier_position=unit.anchor_position,
                scan_heading=(unit.scan_headings[0] + 90) % 360,
                frontier_heading=unit.scan_headings[0],
                newly_known_cells=0,
                confidence_gain=0.0,
                rover_slam=scan_slam,
            )
            parent_id = registry._create_component(
                geometry,
                unknown,
                parent_ids=(parent_id,),
            )
        return registry, parent_id

    def test_low_gain_memory_follows_lineage_and_retires_only_matching_unit(self) -> None:
        scan_slam = line_slam(
            (32, 32), tuple((x, 8) for x in range(5, 13)), version=2,
        )
        registry, child_id = self._two_low_gain_lineage_scans(scan_slam)
        child = registry.components[child_id]
        first_id = child.work_unit_ids[0]

        deferred = registry._apply_low_gain_memory(
            registry._unknown_mask(scan_slam), first_id,
        )

        self.assertEqual(deferred, (first_id,))
        self.assertEqual(
            registry.work_units[first_id].terminal_reason,
            "lineage_low_gain",
        )
        self.assertEqual(child.state, ComponentState.DORMANT)

    def test_new_unknown_support_reenables_low_gain_lineage(self) -> None:
        occupancy = np.full((32, 32), FREE, dtype=np.int8)
        confidence = np.ones((32, 32), dtype=np.float32)
        occupancy[7, 5] = UNKNOWN
        confidence[7, 5] = 0.0
        scan_slam = SlamSnapshot(occupancy, confidence, version=2)
        registry, child_id = self._two_low_gain_lineage_scans(scan_slam)
        child = registry.components[child_id]
        grown_unknown = registry._unknown_mask(scan_slam)
        grown_unknown[9, 5] = True

        deferred = registry._apply_low_gain_memory(
            grown_unknown, child.work_unit_ids[0],
        )

        self.assertEqual(deferred, ())
        self.assertEqual(
            registry.work_units[child.work_unit_ids[0]].state,
            WorkUnitState.READY,
        )

    def test_moved_gateway_is_not_retired_by_old_low_gain_pose(self) -> None:
        scan_slam = line_slam(
            (32, 32), tuple((x, 8) for x in range(5, 13)), version=2,
        )
        registry, child_id = self._two_low_gain_lineage_scans(scan_slam)
        unit_id = registry.components[child_id].work_unit_ids[0]
        registry.work_units[unit_id].anchor_position = (6, 8)

        deferred = registry._apply_low_gain_memory(
            registry._unknown_mask(scan_slam), unit_id,
        )

        self.assertEqual(deferred, ())
        self.assertEqual(registry.work_units[unit_id].state, WorkUnitState.READY)

    def test_one_to_one_continuation_retains_identity(self) -> None:
        registry = self.registry()
        first = tuple((x, 8) for x in range(5, 13))
        second = tuple((x, 9) for x in range(5, 13))

        initial = registry.reconcile(line_slam((32, 32), first, version=1))
        component_id = initial.active_component_ids[0]
        continued = registry.reconcile(line_slam((32, 32), second, version=2))

        self.assertEqual(continued.active_component_ids, (component_id,))
        self.assertTrue(any(
            item.kind == LineageKind.CONTINUED
            and item.child_ids == (component_id,)
            for item in continued.transitions
        ))

    def test_split_creates_children_and_terminal_parent(self) -> None:
        registry = self.registry()
        joined = tuple((x, 8) for x in range(4, 17))
        parent_id = registry.reconcile(
            line_slam((32, 32), joined, version=1)
        ).active_component_ids[0]
        split = tuple((x, 8) for x in (*range(4, 9), *range(12, 17)))
        result = registry.reconcile(line_slam(
            (32, 32),
            split,
            version=2,
            occupied=((9, 8), (10, 8), (11, 8)),
        ))

        transition = next(
            item for item in result.transitions
            if item.kind == LineageKind.SPLIT
        )
        self.assertEqual(transition.parent_ids, (parent_id,))
        self.assertEqual(len(transition.child_ids), 2)
        self.assertEqual(
            registry.components[parent_id].state,
            ComponentState.SPLIT,
        )
        self.assertTrue(all(
            registry.components[item].parent_ids == (parent_id,)
            for item in transition.child_ids
        ))

    def test_merge_creates_new_identity_with_all_parents(self) -> None:
        registry = self.registry()
        split = tuple((x, 8) for x in (*range(4, 9), *range(12, 17)))
        first = registry.reconcile(line_slam(
            (32, 32),
            split,
            version=1,
            occupied=((9, 8), (10, 8), (11, 8)),
        ))
        parents = first.active_component_ids
        joined = tuple((x, 8) for x in range(4, 17))
        merged = registry.reconcile(line_slam((32, 32), joined, version=2))

        transition = next(
            item for item in merged.transitions
            if item.kind == LineageKind.MERGED
        )
        self.assertEqual(transition.parent_ids, parents)
        self.assertEqual(len(transition.child_ids), 1)
        child = registry.components[transition.child_ids[0]]
        self.assertEqual(child.parent_ids, parents)

    def test_wide_zero_gain_retires_only_one_work_unit(self) -> None:
        registry = self.registry(sensor_range=2.0)
        cells = tuple((x, 8) for x in range(2, 25))
        result = registry.reconcile(line_slam((32, 32), cells, version=1))
        component = registry.components[result.active_component_ids[0]]
        units = [registry.work_units[item] for item in component.work_unit_ids]
        self.assertGreater(len(units), 1)
        first = units[0]
        self.assertTrue(registry.claim_work_units((first.work_unit_id,)))
        self.assertTrue(registry.complete_work_unit(
            first.work_unit_id,
            reason="zero_gain",
        ))

        self.assertEqual(first.state, WorkUnitState.VISITED)
        self.assertTrue(any(
            item.state == WorkUnitState.READY for item in units[1:]
        ))
        self.assertEqual(component.state, ComponentState.ACTIVE)

    def test_local_dfs_visit_retires_only_matching_wide_anchor(self) -> None:
        registry = self.registry(sensor_range=2.0)
        cells = tuple((x, 8) for x in range(2, 25))
        initial = registry.reconcile(
            line_slam((32, 32), cells, version=1)
        )
        component_id = initial.active_component_ids[0]
        component = registry.components[component_id]
        units = [registry.work_units[item] for item in component.work_unit_ids]
        self.assertGreater(len(units), 1)

        registry.reconcile(
            line_slam((32, 32), cells, version=2),
            causal_transitions=(CausalTransition(
                predecessor_id=component_id,
                successor_cells=(frozenset(cells),),
                report_id=7,
                visited_successor_anchors=(units[0].anchor_position,),
            ),),
        )

        states = [registry.work_units[item.work_unit_id].state for item in units]
        self.assertEqual(states.count(WorkUnitState.VISITED), 1)
        self.assertGreater(states.count(WorkUnitState.READY), 0)
        self.assertEqual(
            registry.components[component_id].state,
            ComponentState.ACTIVE,
        )

    def test_open_wide_component_is_sweep(self) -> None:
        registry = self.registry(sensor_range=2.0)
        cells = tuple((x, 8) for x in range(2, 25))
        result = registry.reconcile(line_slam((32, 32), cells, version=1))
        component = registry.components[result.active_component_ids[0]]
        self.assertEqual(component.exploration_mode, ExplorationMode.SWEEP)

    def test_sweep_anchors_use_lateral_sensor_footprint(self) -> None:
        registry = self.registry(shape=(128, 128), sensor_range=20.0)
        cells = tuple((x, 64) for x in range(10, 111))

        result = registry.reconcile(
            line_slam((128, 128), cells, version=1)
        )
        component = registry.components[result.active_component_ids[0]]
        units = [
            registry.work_units[unit_id]
            for unit_id in component.work_unit_ids
        ]

        self.assertLessEqual(len(units), 10)
        self.assertEqual(
            set().union(*(unit.cells for unit in units)),
            set(cells),
        )
        self.assertEqual(sum(len(unit.cells) for unit in units), len(cells))

    def test_sustained_wall_contact_is_wall_follow(self) -> None:
        shape = (32, 32)
        occupancy = np.full(shape, OCCUPIED, dtype=np.int8)
        confidence = np.ones(shape, dtype=np.float32)
        occupancy[8, 2:25] = FREE
        occupancy[7, 2:25] = UNKNOWN
        confidence[7, 2:25] = 0.0
        registry = self.registry(shape=shape, sensor_range=2.0)

        result = registry.reconcile(SlamSnapshot(
            occupancy,
            confidence,
            version=1,
        ))
        component = registry.components[result.active_component_ids[0]]

        self.assertEqual(
            component.exploration_mode,
            ExplorationMode.WALL_FOLLOW,
        )

    def test_component_resolves_after_two_missing_authoritative_snapshots(self) -> None:
        registry = self.registry()
        cells = tuple((x, 8) for x in range(5, 13))
        component_id = registry.reconcile(
            line_slam((32, 32), cells, version=1)
        ).active_component_ids[0]
        unknown = line_slam((32, 32), (), version=2)
        first_missing = registry.reconcile(unknown)
        second_missing = registry.reconcile(
            line_slam((32, 32), (), version=3)
        )

        self.assertTrue(any(
            item.kind == LineageKind.DORMANT
            for item in first_missing.transitions
        ))
        self.assertTrue(any(
            item.kind == LineageKind.RESOLVED
            for item in second_missing.transitions
        ))
        self.assertEqual(
            registry.components[component_id].state,
            ComponentState.RESOLVED,
        )


if __name__ == "__main__":
    unittest.main()
