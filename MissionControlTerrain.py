"""Terrain sharing and heatmap helpers for MissionControl.

This mixin groups the distributed-terrain responsibilities that were
previously embedded in MissionControl so the main orchestrator can stay
focused on mission lifecycle and agent startup/shutdown.
"""

import math
import time
from typing import List, Optional, Tuple

import numpy as np
import pygame


class MissionControlTerrainMixin:
    """Mixin that encapsulates terrain exchange and heatmap rendering."""

    def record_terrain_scan(self, samples: List[Tuple[int, int, float, float]]) -> None:
        """Fuse drone terrain observations into the shared known-terrain maps."""
        if not samples:
            return

        map_updated = False
        with self.terrain_lock:
            for x, y, roughness, confidence in samples:
                xi = int(x)
                yi = int(y)
                if yi < 0 or yi >= self.known_roughness.shape[0] or xi < 0 or xi >= self.known_roughness.shape[1]:
                    continue
                if self.map_matrix[yi][xi] != 0:
                    continue

                obs_conf = float(np.clip(confidence, 0.05, 1.0))
                obs_roughness = float(np.clip(roughness, 0.0, 1.0))
                prev_conf = float(self.terrain_confidence[yi, xi])
                prev_value = float(self.known_roughness[yi, xi]) if prev_conf > 0 else obs_roughness
                total_conf = prev_conf + obs_conf
                blended = ((prev_value * prev_conf) + (obs_roughness * obs_conf)) / total_conf

                self.known_roughness[yi, xi] = blended
                self.terrain_confidence[yi, xi] = min(1.0, total_conf)
                map_updated = True

            now = pygame.time.get_ticks() / 1000.0
            if map_updated and (now - self.last_explored_update) >= self.explored_update_interval:
                explored_cells = int(np.count_nonzero(self.terrain_confidence > 0))
                self.control_center.explored_percent = round((explored_cells / self.floor_cells) * 100)
                self.last_explored_update = now
            if map_updated:
                self.presentation.terrain_heatmap_dirty = True

    def toggle_terrain_heatmap(self) -> None:
        """Toggle visibility of the scanned terrain heatmap overlay."""
        self.presentation.toggle_terrain_heatmap()
        if self.presentation.show_terrain_heatmap:
            self.presentation.selected_drone_heatmap_id = None
            for drone in self.drones:
                drone.show_path = False
                drone.show_vision = False
        else:
            for drone in self.drones:
                drone.show_path = True
                drone.show_vision = True

    def toggle_drone_heatmap(self, drone_id: int) -> None:
        """Toggle per-drone heatmap mode for a specific drone id."""
        if drone_id < 0 or drone_id >= len(self.drones):
            return

        self.presentation.toggle_drone_heatmap(drone_id)
        
        if self.presentation.selected_drone_heatmap_id is None:
            for drone in self.drones:
                drone.show_path = True
                drone.show_vision = True
        else:
            self.presentation.show_terrain_heatmap = False
            for drone in self.drones:
                drone.show_path = False
                drone.show_vision = False

    def _has_line_of_sight(self, a: Tuple[int, int], b: Tuple[int, int]) -> bool:
        """Return True when segment a->b does not cross cave walls."""
        x0, y0 = int(a[0]), int(a[1])
        x1, y1 = int(b[0]), int(b[1])

        dx = x1 - x0
        dy = y1 - y0
        steps = max(abs(dx), abs(dy))
        if steps == 0:
            return True

        for i in range(steps + 1):
            t = i / steps
            x = int(round(x0 + dx * t))
            y = int(round(y0 + dy * t))

            if y < 0 or y >= self.map_h or x < 0 or x >= self.map_w:
                return False
            if self.map_matrix[y][x] != 0:
                return False

        return True

    def _maps_differ_enough(
        self,
        source_roughness: np.ndarray,
        source_confidence: np.ndarray,
        target_roughness: np.ndarray,
        target_confidence: np.ndarray
    ) -> bool:
        """Return True when sharing is likely to add meaningful information."""
        stride = max(1, int(self.share_compare_stride))

        src_conf = source_confidence[::stride, ::stride]
        tgt_conf = target_confidence[::stride, ::stride]
        src_rough = source_roughness[::stride, ::stride]
        tgt_rough = target_roughness[::stride, ::stride]
        floor = self.floor_mask[::stride, ::stride]

        src_known = floor & (src_conf > 0.0)
        if not np.any(src_known):
            return False

        tgt_known = floor & (tgt_conf > 0.0)
        src_known_count = int(np.count_nonzero(src_known))
        if src_known_count == 0:
            return False

        new_info = src_known & (~tgt_known)
        new_info_ratio = np.count_nonzero(new_info) / src_known_count
        if new_info_ratio >= self.min_share_new_info_ratio:
            return True

        overlap = src_known & tgt_known
        overlap_count = int(np.count_nonzero(overlap))
        if overlap_count == 0:
            return False

        overlap_delta = np.abs(src_rough - tgt_rough)
        meaningful_delta = overlap & (overlap_delta >= self.min_share_roughness_delta)
        overlap_diff_ratio = np.count_nonzero(meaningful_delta) / overlap_count
        return overlap_diff_ratio >= self.min_share_overlap_diff_ratio

    def _share_terrain_with_nearby_drones(self, drone_id: int) -> None:
        """Check for nearby drones and exchange terrain data."""
        drone = self.drones[drone_id]
        now = time.perf_counter()

        if (now - drone.last_share_time) < drone.share_interval:
            return

        drone.last_share_time = now

        for other_id, other_drone in enumerate(self.drones):
            if other_id == drone_id:
                continue

            pair_key = (min(drone_id, other_id), max(drone_id, other_id))
            if (now - self.last_pair_share.get(pair_key, 0.0)) < self.pair_share_cooldown:
                continue

            dx = drone.pos[0] - other_drone.pos[0]
            dy = drone.pos[1] - other_drone.pos[1]
            distance = math.sqrt(dx * dx + dy * dy)

            proximity_threshold = min(drone.radius, other_drone.radius)
            if distance < 2 * proximity_threshold and self._has_line_of_sight(drone.pos, other_drone.pos):
                with drone.terrain_lock:
                    drone_roughness = drone.known_roughness.copy()
                    drone_confidence = drone.terrain_confidence.copy()
                with drone.exploration_lock:
                    drone_explored_alpha = pygame.surfarray.array_alpha(drone.floor_surf).copy()
                    drone_border = list(drone.border)
                with other_drone.terrain_lock:
                    other_roughness = other_drone.known_roughness.copy()
                    other_confidence = other_drone.terrain_confidence.copy()
                with other_drone.exploration_lock:
                    other_explored_alpha = pygame.surfarray.array_alpha(other_drone.floor_surf).copy()
                    other_border = list(other_drone.border)

                should_other_receive = self._maps_differ_enough(
                    drone_roughness,
                    drone_confidence,
                    other_roughness,
                    other_confidence
                )
                should_drone_receive = self._maps_differ_enough(
                    other_roughness,
                    other_confidence,
                    drone_roughness,
                    drone_confidence
                )
                if not (should_other_receive or should_drone_receive):
                    continue

                if should_other_receive:
                    other_drone.merge_terrain_data(drone_roughness, drone_confidence)
                    other_drone.merge_exploration_data(drone_explored_alpha, drone_border)
                if should_drone_receive:
                    drone.merge_terrain_data(other_roughness, other_confidence)
                    drone.merge_exploration_data(other_explored_alpha, other_border)

                self.presentation.terrain_heatmap_dirty = True
                self.last_pair_share[pair_key] = now

    def _share_terrain_with_rovers(self) -> None:
        """Share terrain knowledge from all drones with rovers."""
        for rover in self.rovers:
            if rover is None:
                continue

            for drone in self.drones:
                dx = rover.pos[0] - drone.pos[0]
                dy = rover.pos[1] - drone.pos[1]
                distance = math.sqrt(dx * dx + dy * dy)

                proximity_threshold = min(rover.radius, drone.radius)
                if distance < proximity_threshold and self._has_line_of_sight(rover.pos, drone.pos):
                    should_rover_receive = self._maps_differ_enough(
                        drone.known_roughness,
                        drone.terrain_confidence,
                        rover.known_roughness,
                        rover.terrain_confidence
                    )
                    if not should_rover_receive:
                        continue

                    with drone.terrain_lock:
                        if not hasattr(rover, 'known_roughness'):
                            rover.known_roughness = drone.known_roughness.copy()
                            rover.terrain_confidence = drone.terrain_confidence.copy()
                        else:
                            h = min(rover.known_roughness.shape[0], drone.known_roughness.shape[0])
                            w = min(rover.known_roughness.shape[1], drone.known_roughness.shape[1])
                            if h > 0 and w > 0:
                                rover_rough = rover.known_roughness[:h, :w]
                                rover_conf = rover.terrain_confidence[:h, :w]
                                drone_rough = np.clip(drone.known_roughness[:h, :w], 0.0, 1.0)
                                drone_conf = np.clip(drone.terrain_confidence[:h, :w], 0.0, 1.0)
                                floor = self.floor_mask[:h, :w]

                                valid = floor & (drone_conf > 0.0)
                                if np.any(valid):
                                    rover_conf_vals = rover_conf[valid]
                                    drone_conf_vals = drone_conf[valid]
                                    rover_rough_vals = rover_rough[valid]
                                    drone_rough_vals = drone_rough[valid]

                                    base_rover = np.where(rover_conf_vals > 0.0, rover_rough_vals, drone_rough_vals)
                                    total = rover_conf_vals + drone_conf_vals
                                    blended = ((base_rover * rover_conf_vals) + (drone_rough_vals * drone_conf_vals)) / np.maximum(total, 1e-6)

                                    rover_rough[valid] = blended
                                    rover_conf[valid] = np.minimum(1.0, total)

                    self.presentation.terrain_heatmap_dirty = True

    def _refresh_terrain_heatmap(self, drone_id: Optional[int] = None) -> None:
        """Rebuild the cached terrain heatmap surface."""
        if drone_id is not None and 0 <= drone_id < len(self.drones):
            drone = self.drones[drone_id]
            with drone.terrain_lock:
                roughness = np.clip(drone.known_roughness.copy(), 0.0, 1.0)
                confidence = np.clip(drone.terrain_confidence.copy(), 0.0, 1.0)
            self._render_heatmap_from_maps(roughness, confidence)
            return

        entity_maps: List[Tuple[np.ndarray, np.ndarray]] = []

        for drone in self.drones:
            with drone.terrain_lock:
                entity_maps.append((drone.known_roughness.copy(), drone.terrain_confidence.copy()))

        for rover in self.rovers:
            if rover is not None and hasattr(rover, 'known_roughness') and hasattr(rover, 'terrain_confidence'):
                entity_maps.append((rover.known_roughness.copy(), rover.terrain_confidence.copy()))

        if not entity_maps:
            self.terrain_heatmap_surf.fill((0, 0, 0, 0))
            self.terrain_heatmap_dirty = False
            return

        h, w = self.floor_mask.shape
        common_known = self.floor_mask.copy()
        weighted_num = np.zeros((h, w), dtype=np.float32)
        weighted_den = np.zeros((h, w), dtype=np.float32)
        min_conf = np.ones((h, w), dtype=np.float32)

        for roughness_map, confidence_map in entity_maps:
            eh = min(h, roughness_map.shape[0], confidence_map.shape[0])
            ew = min(w, roughness_map.shape[1], confidence_map.shape[1])
            if eh <= 0 or ew <= 0:
                self.terrain_heatmap_surf.fill((0, 0, 0, 0))
                self.terrain_heatmap_dirty = False
                return

            entity_conf = np.zeros((h, w), dtype=np.float32)
            entity_rough = np.zeros((h, w), dtype=np.float32)
            entity_conf[:eh, :ew] = np.clip(confidence_map[:eh, :ew], 0.0, 1.0)
            entity_rough[:eh, :ew] = np.clip(roughness_map[:eh, :ew], 0.0, 1.0)

            known = entity_conf > 0.0
            common_known &= known
            weighted_num += entity_rough * entity_conf
            weighted_den += entity_conf
            min_conf = np.minimum(min_conf, entity_conf)

        roughness = np.zeros((h, w), dtype=np.float32)
        confidence = np.zeros((h, w), dtype=np.float32)
        valid = common_known
        if np.any(valid):
            roughness[valid] = weighted_num[valid] / np.maximum(weighted_den[valid], 1e-6)
            confidence[valid] = min_conf[valid]

        self._render_heatmap_from_maps(roughness, confidence)

    def _render_heatmap_from_maps(self, roughness: np.ndarray, confidence: np.ndarray) -> None:
        """Render a heatmap surface from roughness and confidence arrays."""
        valid_mask = confidence > 0.0
        if not np.any(valid_mask):
            self.presentation.terrain_heatmap_surf.fill((0, 0, 0, 0))
            self.presentation.terrain_heatmap_dirty = False
            return

        ramp = np.clip(((roughness - 0.5) * 1.8) + 0.5, 0.0, 1.0)
        band = np.clip((ramp * 5.0).astype(np.int8), 0, 4)

        red = np.zeros_like(ramp, dtype=np.float32)
        green = np.zeros_like(ramp, dtype=np.float32)
        blue = np.zeros_like(ramp, dtype=np.float32)

        red[band == 0], green[band == 0], blue[band == 0] = 30.0, 80.0, 235.0
        red[band == 1], green[band == 1], blue[band == 1] = 45.0, 190.0, 70.0
        red[band == 2], green[band == 2], blue[band == 2] = 245.0, 225.0, 60.0
        red[band == 3], green[band == 3], blue[band == 3] = 245.0, 145.0, 40.0
        red[band == 4], green[band == 4], blue[band == 4] = 235.0, 45.0, 40.0

        alpha = np.where(valid_mask, 35.0 + (confidence * 125.0), 0.0)

        red = np.clip(red, 0.0, 255.0).astype(np.uint8)
        green = np.clip(green, 0.0, 255.0).astype(np.uint8)
        blue = np.clip(blue, 0.0, 255.0).astype(np.uint8)
        alpha = np.clip(alpha, 0.0, 160.0).astype(np.uint8)

        rgb_view = pygame.surfarray.pixels3d(self.presentation.terrain_heatmap_surf)
        alpha_view = pygame.surfarray.pixels_alpha(self.presentation.terrain_heatmap_surf)
        rgb_view[:, :, 0] = red.T
        rgb_view[:, :, 1] = green.T
        rgb_view[:, :, 2] = blue.T
        alpha_view[:, :] = alpha.T
        del rgb_view
        del alpha_view

        self.presentation.terrain_heatmap_dirty = False

    def draw_terrain_heatmap(self) -> None:
        """Blit the terrain heatmap overlay when a heatmap mode is enabled."""
        if not self.presentation.show_terrain_heatmap and self.presentation.selected_drone_heatmap_id is None:
            return
        self._refresh_terrain_heatmap(self.presentation.selected_drone_heatmap_id)
        self.game.window.blit(self.presentation.terrain_heatmap_surf, (0, 0))

    def acquire_rover_target(self, rover_id: int, current_pos: Tuple[int, int]) -> Optional[Tuple[int, int]]:
        """Choose and reserve a discovered rough-terrain target for a rover."""
        with self.rover_assignment_lock, self.terrain_lock:
            assigned_targets = {target for rid, target in self.rover_assignments.items() if rid != rover_id}
            candidate_mask = (
                (np.asarray(self.map_matrix) == 0)
                & (self.terrain_confidence >= 0.25)
                & (self.known_roughness >= 0.35)
            )

            if not np.any(candidate_mask):
                return None

            ys, xs = np.where(candidate_mask)
            best_target = None
            best_score = float('-inf')
            norm = max(1.0, math.hypot(self.game.width, self.game.height))

            for x, y in zip(xs, ys):
                target = (int(x), int(y))
                if target in assigned_targets or target in self.completed_rover_targets:
                    continue

                distance_penalty = math.dist(current_pos, target) / norm
                score = (0.7 * float(self.known_roughness[y, x])) + (0.3 * float(self.terrain_confidence[y, x])) - distance_penalty
                if score > best_score:
                    best_score = score
                    best_target = target

            if best_target is not None:
                self.rover_assignments[rover_id] = best_target
            return best_target

    def release_rover_target(self, rover_id: int, completed: bool = False) -> None:
        """Release or mark complete a rover terrain target reservation."""
        with self.rover_assignment_lock:
            target = self.rover_assignments.pop(rover_id, None)
            if completed and target is not None:
                self.completed_rover_targets.add(target)
