"""Shared occupancy-frontier mask and significance operations."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from mapping.slam_map import FREE, UNKNOWN


Position = tuple[int, int]


@dataclass(frozen=True)
class FrontierComponentDiagnostic:
    """Spatial identity and disposition of one raw frontier component."""

    component_id: int
    size: int
    bounding_box: tuple[int, int, int, int]
    centroid: tuple[float, float]
    unknown_basin_ids: tuple[int, ...]
    maximum_unknown_support_cells: int
    maximum_rescuable_unknown_support_cells: int
    touches_border_connected_unknown: bool
    retained: bool
    retention_reason: str


@dataclass(frozen=True)
class FrontierFilterDiagnostics:
    """Summary of one significance-filter pass over a SLAM snapshot."""

    raw_frontier_pixels: int
    raw_component_count: int
    component_sizes: tuple[int, ...]
    significant_frontier_pixels: int
    significant_component_count: int
    large_component_count: int
    unknown_supported_component_count: int
    unknown_supported_candidate_count: int
    redundant_unknown_supported_component_count: int
    frontier_unknown_basin_count: int
    significant_unknown_basin_count: int
    border_connected_unknown_basin_count: int
    border_connected_unknown_supported_candidate_count: int
    discarded_component_count: int
    discarded_frontier_pixels: int
    minimum_component_cells: int
    minimum_unknown_support_cells: int
    components: tuple[FrontierComponentDiagnostic, ...]


def eight_neighbor_adjacency(mask: np.ndarray) -> np.ndarray:
    """Return cells adjacent to at least one true eight-neighbor."""
    height, width = mask.shape
    adjacent = np.zeros_like(mask, dtype=bool)
    for offset_y in (-1, 0, 1):
        for offset_x in (-1, 0, 1):
            if offset_x == 0 and offset_y == 0:
                continue
            source_y = slice(
                max(0, -offset_y),
                height - max(0, offset_y),
            )
            target_y = slice(
                max(0, offset_y),
                height - max(0, -offset_y),
            )
            source_x = slice(
                max(0, -offset_x),
                width - max(0, offset_x),
            )
            target_x = slice(
                max(0, offset_x),
                width - max(0, -offset_x),
            )
            adjacent[target_y, target_x] |= mask[source_y, source_x]
    return adjacent


def known_free_frontier_mask(
    occupancy: np.ndarray,
    confidence: np.ndarray,
    confidence_threshold: float,
) -> np.ndarray:
    """Return confident free cells bordering unknown SLAM evidence."""
    threshold = float(confidence_threshold)
    known_free = (occupancy == FREE) & (confidence >= threshold)
    unknown = (occupancy == UNKNOWN) | (confidence < threshold)
    return known_free & eight_neighbor_adjacency(unknown)


def eight_connected_components(
    mask: np.ndarray,
) -> tuple[tuple[Position, ...], ...]:
    """Return deterministic eight-connected true-cell components."""
    if mask.ndim != 2:
        raise ValueError("frontier masks must be two-dimensional")

    pixel_y, pixel_x = np.nonzero(mask)
    remaining = {
        (int(x), int(y))
        for y, x in zip(pixel_y, pixel_x)
    }
    components: list[tuple[Position, ...]] = []
    while remaining:
        seed = remaining.pop()
        stack = [seed]
        component: list[Position] = []
        while stack:
            point = stack.pop()
            component.append(point)
            x, y = point
            for offset_y in (-1, 0, 1):
                for offset_x in (-1, 0, 1):
                    if offset_x == 0 and offset_y == 0:
                        continue
                    neighbor = (x + offset_x, y + offset_y)
                    if neighbor in remaining:
                        remaining.remove(neighbor)
                        stack.append(neighbor)
        components.append(tuple(sorted(
            component,
            key=lambda point: (point[1], point[0]),
        )))
    return tuple(sorted(
        components,
        key=lambda component: (
            -len(component),
            component[0][1],
            component[0][0],
        ),
    ))


def significant_frontier_mask(
    occupancy: np.ndarray,
    confidence: np.ndarray,
    confidence_threshold: float,
    *,
    minimum_component_cells: int,
    minimum_unknown_support_cells: int,
) -> tuple[np.ndarray, FrontierFilterDiagnostics]:
    """Keep useful frontier components and discard isolated pixel residue.

    A component is useful when it is large enough to be visually meaningful,
    or when one of its unknown neighbors opens into enough connected unknown
    space.  At most one small doorway is retained for each connected unknown
    basin; otherwise hundreds of isolated frontier specks can all claim the
    same unexplored area as independent work.
    """
    if occupancy.shape != confidence.shape or occupancy.ndim != 2:
        raise ValueError("occupancy and confidence must be matching 2D arrays")
    minimum_cells = int(minimum_component_cells)
    minimum_unknown = int(minimum_unknown_support_cells)
    if minimum_cells <= 0:
        raise ValueError("minimum_component_cells must be positive")
    if minimum_unknown <= 0:
        raise ValueError("minimum_unknown_support_cells must be positive")

    threshold = float(confidence_threshold)
    raw = known_free_frontier_mask(
        occupancy,
        confidence,
        threshold,
    )
    components = eight_connected_components(raw)
    unknown = (occupancy == UNKNOWN) | (confidence < threshold)
    basin_count, basin_labels = cv2.connectedComponents(
        unknown.astype(np.uint8),
        connectivity=8,
    )
    basin_sizes = np.bincount(
        basin_labels.ravel(),
        minlength=basin_count,
    )
    if basin_sizes.size:
        basin_sizes[0] = 0

    component_basins = tuple(
        _adjacent_unknown_basins(component, basin_labels)
        for component in components
    )
    border_connected_basins = _border_connected_basin_ids(basin_labels)
    size_qualifying_basins = tuple(
        tuple(
            basin_id
            for basin_id in basin_ids
            if int(basin_sizes[basin_id]) >= minimum_unknown
        )
        for basin_ids in component_basins
    )
    qualifying_basins = tuple(
        tuple(
            basin_id
            for basin_id in basin_ids
            if basin_id not in border_connected_basins
        )
        for basin_ids in size_qualifying_basins
    )
    frontier_basin_ids = {
        basin_id
        for basin_ids in component_basins
        for basin_id in basin_ids
    }

    significant = np.zeros_like(raw, dtype=bool)
    large_count = 0
    supported_count = 0
    supported_candidate_count = 0
    redundant_supported_count = 0
    border_supported_candidate_count = 0
    discarded_pixels = 0
    represented_basins: set[int] = set()
    retained_basin_ids: set[int] = set()
    component_diagnostics: list[FrontierComponentDiagnostic] = []
    for component_id, (
        component,
        basin_ids,
        size_supported_basins,
        supported_basins,
    ) in enumerate(
        zip(
            components,
            component_basins,
            size_qualifying_basins,
            qualifying_basins,
        )
    ):
        border_supported_basins = tuple(
            basin_id
            for basin_id in size_supported_basins
            if basin_id in border_connected_basins
        )
        if len(component) >= minimum_cells:
            retained = True
            reason = "large_component"
            large_count += 1
            represented_basins.update(supported_basins)
        elif supported_basins:
            supported_candidate_count += 1
            novel_basins = tuple(
                basin_id
                for basin_id in supported_basins
                if basin_id not in represented_basins
            )
            retained = bool(novel_basins)
            if retained:
                reason = "unknown_basin_gateway"
                supported_count += 1
                represented_basins.update(supported_basins)
            else:
                reason = "duplicate_unknown_basin_gateway"
                redundant_supported_count += 1
        elif border_supported_basins:
            retained = False
            reason = "border_connected_unknown_basin"
            border_supported_candidate_count += 1
        else:
            retained = False
            reason = "insufficient_unknown_support"
        if retained:
            retained_basin_ids.update(basin_ids)
            for x, y in component:
                significant[y, x] = True
        else:
            discarded_pixels += len(component)
        component_diagnostics.append(_component_diagnostic(
            component_id,
            component,
            basin_ids,
            basin_sizes,
            border_connected_basins,
            retained=retained,
            retention_reason=reason,
        ))

    significant_count = large_count + supported_count
    diagnostics = FrontierFilterDiagnostics(
        raw_frontier_pixels=int(np.count_nonzero(raw)),
        raw_component_count=len(components),
        component_sizes=tuple(len(component) for component in components),
        significant_frontier_pixels=int(np.count_nonzero(significant)),
        significant_component_count=significant_count,
        large_component_count=large_count,
        unknown_supported_component_count=supported_count,
        unknown_supported_candidate_count=supported_candidate_count,
        redundant_unknown_supported_component_count=(
            redundant_supported_count
        ),
        frontier_unknown_basin_count=len(frontier_basin_ids),
        significant_unknown_basin_count=len(retained_basin_ids),
        border_connected_unknown_basin_count=len(
            frontier_basin_ids & border_connected_basins
        ),
        border_connected_unknown_supported_candidate_count=(
            border_supported_candidate_count
        ),
        discarded_component_count=len(components) - significant_count,
        discarded_frontier_pixels=discarded_pixels,
        minimum_component_cells=minimum_cells,
        minimum_unknown_support_cells=minimum_unknown,
        components=tuple(component_diagnostics),
    )
    return significant, diagnostics


def _border_connected_basin_ids(basin_labels: np.ndarray) -> set[int]:
    """Return unknown-basin labels connected to the map boundary."""
    if basin_labels.size == 0:
        return set()
    border_values = np.concatenate((
        basin_labels[0, :],
        basin_labels[-1, :],
        basin_labels[:, 0],
        basin_labels[:, -1],
    ))
    return {
        int(value)
        for value in np.unique(border_values)
        if int(value) > 0
    }


def frontier_neighborhood_masks(
    unknown: np.ndarray,
    components: tuple[frozenset[Position], ...],
    *,
    halo: int,
) -> tuple[np.ndarray, ...]:
    """Freeze component halos and adjacent enclosed basins for one epoch.

    Basin labels are snapshot-local geometry, never suppression identities.
    Border-connected unknown space cannot expand a component's bounded halo.
    """
    if unknown.ndim != 2 or halo < 0:
        raise ValueError("neighborhoods require a 2D mask and nonnegative halo")
    _, labels = cv2.connectedComponents(
        unknown.astype(np.uint8), connectivity=8,
    )
    border_ids = _border_connected_basin_ids(labels)
    kernel = np.ones((2 * halo + 1, 2 * halo + 1), dtype=np.uint8)
    masks = []
    for component in components:
        pixels = np.zeros(unknown.shape, dtype=np.uint8)
        for x, y in component:
            pixels[y, x] = 1
        scope = cv2.dilate(pixels, kernel).astype(bool)
        basin_ids = set(_adjacent_unknown_basins(tuple(component), labels))
        enclosed = np.isin(labels, tuple(basin_ids - border_ids))
        scope |= enclosed | eight_neighbor_adjacency(enclosed)
        scope.setflags(write=False)
        masks.append(scope)
    return tuple(masks)


def _adjacent_unknown_basins(
    frontier_component: tuple[Position, ...],
    basin_labels: np.ndarray,
) -> tuple[int, ...]:
    """Return connected-unknown labels touched by a frontier component."""
    height, width = basin_labels.shape
    basin_ids: set[int] = set()
    for x, y in frontier_component:
        top = max(0, y - 1)
        bottom = min(height, y + 2)
        left = max(0, x - 1)
        right = min(width, x + 2)
        basin_ids.update(
            int(value)
            for value in np.unique(
                basin_labels[top:bottom, left:right]
            )
            if int(value) > 0
        )
    return tuple(sorted(basin_ids))


def _component_diagnostic(
    component_id: int,
    component: tuple[Position, ...],
    basin_ids: tuple[int, ...],
    basin_sizes: np.ndarray,
    border_connected_basins: set[int],
    *,
    retained: bool,
    retention_reason: str,
) -> FrontierComponentDiagnostic:
    """Build a compact trace-safe component description."""
    xs = tuple(point[0] for point in component)
    ys = tuple(point[1] for point in component)
    return FrontierComponentDiagnostic(
        component_id=int(component_id),
        size=len(component),
        bounding_box=(min(xs), min(ys), max(xs), max(ys)),
        centroid=(
            sum(xs) / len(xs),
            sum(ys) / len(ys),
        ),
        unknown_basin_ids=basin_ids,
        maximum_unknown_support_cells=max(
            (int(basin_sizes[basin_id]) for basin_id in basin_ids),
            default=0,
        ),
        maximum_rescuable_unknown_support_cells=max(
            (
                int(basin_sizes[basin_id])
                for basin_id in basin_ids
                if basin_id not in border_connected_basins
            ),
            default=0,
        ),
        touches_border_connected_unknown=any(
            basin_id in border_connected_basins
            for basin_id in basin_ids
        ),
        retained=bool(retained),
        retention_reason=str(retention_reason),
    )
