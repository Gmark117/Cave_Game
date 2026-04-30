# Detailed Implementation Checklist: All Phases

---

## PHASE 1: Distributed Data Sharing (POI & Path Extension)

### 1.1 Add Data Structures to Drone

**File**: Drone.py — Import section + `__init__`

- [ ] **Add imports** (top of file):
  ```python
  from dataclasses import dataclass, field
  from typing import Dict, Tuple, List
  ```

- [ ] **Add POI dataclass** (before Drone class, ~line 20):
  ```python
  @dataclass
  class POI:
      location: Tuple[int, int]
      poi_type: str  # "victim", "landmark", "resource"
      discovered_by: int  # drone_id who found it
      timestamp: float
      confidence: float = 1.0
  ```

- [ ] **Add to Drone.__init__** (after line 85, before return):
  ```python
  self.known_pois: Dict[str, List[POI]] = {}  # type -> list
  self.known_paths: Dict[Tuple[int, int], List[Tuple[int, int]]] = {}  # target -> path
  ```

### 1.2 Add POI Merge Method to Drone

**File**: Drone.py — After `merge_terrain_data()` method

- [ ] **Add `merge_poi_data()` method** (after line 515):
  ```python
  def merge_poi_data(self, other_pois: Dict[str, List[POI]]) -> None:
      """Merge POI observations from another drone using confidence weighting."""
      with self.terrain_lock:
          for poi_type, other_list in other_pois.items():
              if poi_type not in self.known_pois:
                  self.known_pois[poi_type] = []
              
              for other_poi in other_list:
                  # Check if nearby POI already known (within 20px)
                  existing = next(
                      (p for p in self.known_pois[poi_type] 
                       if math.dist(p.location, other_poi.location) < 20),
                      None
                  )
                  if existing:
                      # Increase confidence on repeated discovery
                      existing.confidence = min(1.0, existing.confidence + other_poi.confidence * 0.5)
                      existing.discovered_by = other_poi.discovered_by
                  else:
                      self.known_pois[poi_type].append(other_poi)
  ```

### 1.3 Initialize Global POI Storage in MissionControl

**File**: MissionControl.py — `__init__` method

- [ ] **Add after line 70** (after terrain sharing setup):
  ```python
  # Global POI and path sharing
  self.global_pois: Dict[str, List[POI]] = {}
  self.global_paths_cache: Dict[str, Dict] = {}  # uuid -> {"path": [...], "roughness": [...]}
  ```

### 1.4 Add POI Sharing Logic to MissionControlTerrain

**File**: MissionControlTerrain.py — Add new method

- [ ] **Add `_share_pois_with_nearby_drones()` method** (after `_share_terrain_with_nearby_drones()`):
  ```python
  def _share_pois_with_nearby_drones(self, drone_id: int) -> None:
      """Exchange POI discoveries with drones within proximity (distance < 200px and LOS)."""
      drone = self.drones[drone_id]
      if not drone.known_pois:
          return
      
      for other_id, other_drone in enumerate(self.drones):
          if other_id == drone_id or not other_drone.active:
              continue
          
          distance = math.dist(drone.pos, other_drone.pos)
          if distance > 200 or not self._has_line_of_sight(drone.pos, other_drone.pos):
              continue
          
          # Exchange POI data
          other_drone.merge_poi_data(deepcopy(drone.known_pois))
          drone.merge_poi_data(deepcopy(other_drone.known_pois))
  ```

- [ ] **Call POI sharing in drone_thread()** — Modify MissionControl.py, add after `_share_terrain_with_nearby_drones()`:
  ```python
  self._share_pois_with_nearby_drones(drone_id)  # NEW: Share POI discoveries
  ```

### 1.5 Validation Checklist

- [ ] Both exploration and S&R missions run without import errors
- [ ] Drones initialize with empty `known_pois` and `known_paths` dicts
- [ ] When drones pass within 200px, POI data is exchanged
- [ ] POI confidence increases when rediscovered
- [ ] Control Center displays discovered POIs (defer full UI to Phase 5)

---

## PHASE 2: Waypoint Path Segmentation

### 2.1 Create WaypointPlanner Module

**File**: Create WaypointPlanner.py (new file)

- [ ] **Create file with:**
  ```python
  from dataclasses import dataclass
  from typing import Tuple, List
  import numpy as np
  import math
  from collections import defaultdict
  
  @dataclass
  class Waypoint:
      location: Tuple[int, int]
      frontier_density: float  # 0-1, higher = more unexplored nearby
      roughness: float
      priority: int  # 0 = highest priority
  
  class WaypointPlanner:
      def __init__(self, map_h: int, map_w: int, grid_spacing: int = 120):
          self.map_h = map_h
          self.map_w = map_w
          self.grid_spacing = grid_spacing
          self.waypoints: List[Waypoint] = []
      
      def generate_waypoints(self, goal: Tuple[int, int], 
                            frontier_cells: List[Tuple[int, int]],
                            roughness_map: np.ndarray) -> List[Tuple[int, int]]:
          """Cluster frontier cells into a grid of waypoints, sorted by value."""
          if not frontier_cells:
              return [goal]
          
          # Divide map into grid cells
          grid_cells = defaultdict(list)
          for fx, fy in frontier_cells:
              grid_x = fx // self.grid_spacing
              grid_y = fy // self.grid_spacing
              grid_cells[(grid_x, grid_y)].append((fx, fy))
          
          # Score each grid cell
          waypoints = []
          for (gx, gy), cells in grid_cells.items():
              # Centroid of frontier points in this grid cell
              cx = sum(x for x, y in cells) / len(cells)
              cy = sum(y for x, y in cells) / len(cells)
              
              # Score: frontier density (number of cells)
              frontier_score = len(cells)
              
              # Average roughness in region
              roughness = np.mean([roughness_map[int(y)][int(x)] for x, y in cells])
              
              # Distance to goal (closer = lower priority but more realistic)
              distance_to_goal = math.dist((cx, cy), goal)
              
              waypoint = Waypoint(
                  location=(int(cx), int(cy)),
                  frontier_density=frontier_score,
                  roughness=roughness,
                  priority=0
              )
              waypoints.append(waypoint)
          
          # Sort by frontier density (descending) and roughness (ascending)
          waypoints.sort(key=lambda w: (-w.frontier_density, w.roughness))
          
          # Keep top 5 waypoints to avoid excessive segmentation
          waypoints = waypoints[:5]
          
          self.waypoints = waypoints
          return [w.location for w in waypoints]
  ```

### 2.2 Initialize WaypointPlanner in MissionControl

**File**: MissionControl.py — `__init__` method

- [ ] **Add import** (top of file):
  ```python
  from WaypointPlanner import WaypointPlanner
  ```

- [ ] **Add initialization** (after terrain initialization, ~line 75):
  ```python
  self.waypoint_planner = WaypointPlanner(self.map_h, self.map_w, grid_spacing=120)
  ```

### 2.3 Modify Rover Path Computation with Waypoints

**File**: MissionControl.py — `compute_rover_path()` method

- [ ] **Replace entire method** with:
  ```python
  def compute_rover_path_with_waypoints(self, start: Tuple[int, int], 
                                        goal: Tuple[int, int]) -> List[Tuple[int, int]]:
      """Compute rover path segmented through waypoints for long distances."""
      direct_distance = math.dist(start, goal)
      
      # Use direct path for short distances
      if direct_distance < 300:
          path = AStarPathfinder.compute_weighted_path(
              self.map_matrix, self.known_roughness, 
              self.terrain_confidence, start, goal
          )
          return path if path else [goal]
      
      # For long distances, use waypoint segmentation
      frontier_cells = [tuple(pos) for pos, val in np.ndenumerate(self.terrain_confidence) 
                        if val < 0.5]
      waypoints = self.waypoint_planner.generate_waypoints(goal, frontier_cells, self.known_roughness)
      
      full_path = []
      current = start
      
      for wp in waypoints:
          segment = AStarPathfinder.compute_weighted_path(
              self.map_matrix, self.known_roughness,
              self.terrain_confidence, current, wp
          )
          if segment:
              full_path.extend(segment[1:])  # Skip duplicate start
              current = wp
      
      # Final segment to actual goal
      final_segment = AStarPathfinder.compute_weighted_path(
          self.map_matrix, self.known_roughness,
          self.terrain_confidence, current, goal
      )
      if final_segment:
          full_path.extend(final_segment[1:])
      
      return full_path if full_path else [goal]
  ```

- [ ] **Update call site** — Find where `compute_rover_path()` is called in rover initialization and change to:
  ```python
  # OLD: self.current_path = self.control.compute_rover_path(self.pos, target)
  # NEW:
  self.current_path = self.control.compute_rover_path_with_waypoints(self.pos, target)
  ```

### 2.4 Validation Checklist

- [ ] WaypointPlanner.py imports without errors
- [ ] Rover initializes with waypoint-segmented paths for distances > 300px
- [ ] Short paths still use direct A* routing
- [ ] Rover follows segmented paths correctly (popping waypoints in order)
- [ ] Run with large map (>1000px) and confirm waypoints reduce path oscillation

---

## PHASE 3: Battery Management

### 3.1 Modify Drone.move() for Battery Consumption

**File**: Drone.py — `move()` method, end

- [ ] **Add after existing position update logic** (after reaching any goal/border):
  ```python
  # Battery consumption per movement
  self.battery = max(0, self.battery - 0.1)  # 0.1 per move (1000 moves = full discharge)
  
  # Return to base if battery critical
  if self.battery < 10 and not self.returning_home:
      self.returning_home = True
  ```

### 3.2 Modify Drone.scan_terrain() for Battery Consumption

**File**: Drone.py — `scan_terrain()` method, end

- [ ] **Add at end of method** (after all scan logic):
  ```python
  # Battery consumption for scanning
  self.battery = max(0, self.battery - 0.05)  # Cheaper than movement
  ```

### 3.3 Modify Drone.mission_completed() for Battery Checks

**File**: Drone.py — `mission_completed()` method

- [ ] **Replace entire method** with:
  ```python
  def mission_completed(self) -> bool:
      """Check if drone has completed its mission considering battery."""
      
      # FAIL: Battery depleted
      if self.battery <= 0:
          if not self.done:
              self.done = True
              self.returning_home = False
              print(f"Drone {self.id} battery depleted - mission failed")
          return True
      
      # INCOMPLETE: Not yet explored
      if not self.explored:
          return False
      
      # EXPLORATION MODE: Check if border exhausted and return to start
      if not self.border and not self.done:
          self.returning_home = True
          return False
      
      # RECHARGE AT BASE: Battery refilled when at start position
      if self.done and math.dist(self.pos, self.start_pos) < 20:
          self.battery = 100  # Recharge at base
          print(f"Drone {self.id} recharged at base")
          return True
      
      return self.done
  ```

### 3.4 Update ControlCenter Battery Display

**File**: ControlCenter.py — Battery rendering section

- [ ] **Find battery display loop** (around line 330-335) and modify colors:
  ```python
  battery_percent = int((drone.battery / 100) * 100)
  
  # Color based on battery level
  if battery_percent <= 0:
      battery_color = pygame.Color("red")
  elif battery_percent < 20:
      battery_color = pygame.Color("red")  # Critical
  elif battery_percent < 50:
      battery_color = pygame.Color("yellow")  # Warning
  else:
      battery_color = pygame.Color("green")  # Healthy
  
  # Existing rendering code but with battery_color
  ```

### 3.5 Validation Checklist

- [ ] Drones initialize with battery = 100
- [ ] Battery decreases by ~0.1 per move, ~0.05 per scan
- [ ] Drone returns home when battery < 10
- [ ] Battery recharges to 100 at start_pos
- [ ] Mission fails if drone battery hits 0 before returning
- [ ] Run 1000+ move test; battery should deplete in ~10000 moves
- [ ] UI shows battery in green (>50), yellow (20-50), red (<20)

---

## PHASE 4: Non-Random Exploration

### 4.1 Add Frontier Scoring Method to Drone

**File**: Drone.py — After `reach_border()` method

- [ ] **Add `select_best_frontier_direction()` method** (before `find_new_node()`):
  ```python
  def select_best_frontier_direction(self, valid_targets: List[Tuple[int, int]]) -> Tuple[int, int]:
      """Score frontier candidates by proximity, novelty, and exploration value."""
      
      if not valid_targets:
          return None
      
      best_target = None
      best_score = -float('inf')
      
      for target in valid_targets:
          # 1. Proximity bonus (closer = better)
          dist_score = -math.dist(self.pos, target)
          
          # 2. Frontier density (unexplored cells near target)
          nearby_frontier = sum(
              1 for b in self.border 
              if math.dist(b, target) < self.radius
          )
          frontier_score = nearby_frontier
          
          # 3. Novelty penalty (penalize revisited cells)
          # Track via direction log - cells we've been near before get penalized
          x_int, y_int = int(target[0]), int(target[1])
          revisit_count = 0
          for prev_x, prev_y in self.dir_log[-50:]:  # Check last 50 moves
              if abs(prev_x - x_int) < self.radius and abs(prev_y - y_int) < self.radius:
                  revisit_count += 1
          revisit_penalty = -revisit_count * 0.3
          
          # 4. Direction continuity (prefer less sudden turns)
          angle_to_target = math.atan2(target[1] - self.pos[1], target[0] - self.pos[0])
          direction_continuity = -abs(self._normalize_angle(angle_to_target - self.current_angle))
          
          # Weighted combination
          total_score = (
              dist_score * 1.0 +              # Proximity: weight 1.0
              frontier_score * 0.5 +          # Frontier density: weight 0.5
              revisit_penalty * 0.3 +         # Novelty: weight 0.3
              direction_continuity * 0.1      # Continuity: weight 0.1
          )
          
          if total_score > best_score:
              best_score = total_score
              best_target = target
      
      return best_target if best_target else valid_targets[0]
  
  def _normalize_angle(self, angle: float) -> float:
      """Normalize angle to [-pi, pi]."""
      while angle > math.pi:
          angle -= 2 * math.pi
      while angle < -math.pi:
          angle += 2 * math.pi
      return angle
  ```

### 4.2 Modify find_new_node() to Use Scoring

**File**: Drone.py — `find_new_node()` method

- [ ] **Find line ~210** (random direction selection):
  ```python
  # OLD CODE:
  # self.dir = rand.choice(valid_dirs)
  
  # NEW CODE:
  chosen_target = self.select_best_frontier_direction(valid_targets)
  if chosen_target:
      # Find the direction index that corresponds to this target
      for i, target in enumerate(valid_targets):
          if target == chosen_target:
              self.dir = valid_dirs[i]
              target_cell = chosen_target
              break
      else:
          # Fallback if target not found in list
          self.dir = valid_dirs[0]
          target_cell = valid_targets[0] if valid_targets else None
  else:
      # Fallback: random selection if scoring fails
      self.dir = rand.choice(valid_dirs)
      target_cell = rand.choice(valid_targets)
  ```

### 4.3 Initialize Direction Tracking in Drone.__init__

**File**: Drone.py — `__init__` method

- [ ] **Add tracking structures** (after exploration state init):
  ```python
  self.dir_log: List[Tuple[int, int]] = []  # History of positions visited
  self.current_angle: float = 0.0  # Current heading in radians
  ```

### 4.4 Update Direction Log in move()

**File**: Drone.py — `move()` method

- [ ] **Add after position update**:
  ```python
  # Track visited positions for novelty heuristic
  self.dir_log.append((int(self.pos[0]), int(self.pos[1])))
  if len(self.dir_log) > 200:  # Keep last 200 positions
      self.dir_log.pop(0)
  
  # Update current angle based on movement
  if len(self.dir_log) > 1:
      dx = self.dir_log[-1][0] - self.dir_log[-2][0]
      dy = self.dir_log[-1][1] - self.dir_log[-2][1]
      if dx or dy:
          self.current_angle = math.atan2(dy, dx)
  ```

### 4.5 Validation Checklist

- [ ] Drones select frontier directions based on scoring (not random)
- [ ] Frontier density drives direction selection (higher frontier = higher score)
- [ ] Revisited areas are penalized (drones avoid loops)
- [ ] Run 500+ moves and confirm drones explore systematically vs wandering
- [ ] Verify no new errors in boundary selection logic

---

## PHASE 5: Search & Rescue Mission

### 5.1 Create TargetEntity Dataclass

**File**: Drone.py — Top of file with other dataclasses

- [ ] **Add import** (if not present):
  ```python
  from dataclasses import dataclass, field
  ```

- [ ] **Add TargetEntity after POI dataclass**:
  ```python
  @dataclass
  class TargetEntity:
      id: str  # UUID or simple name "target_0"
      location: Tuple[int, int]
      discovered_by: List[int] = field(default_factory=list)  # Drone IDs
      discovered_time: float = 0.0
      rescued: bool = False
      rescued_by: Optional[int] = None
      rescue_time: float = 0.0
  ```

### 5.2 Initialize Targets in MissionControl

**File**: MissionControl.py — `__init__` method

- [ ] **Add import** (top of file):
  ```python
  from Drone import TargetEntity  # Import target entity
  ```

- [ ] **Add target initialization** (after mission assignment, ~line 76):
  ```python
  self.targets: List[TargetEntity] = []
  
  if self.mission == 1:  # Search and Rescue mission
      # Generate random target locations in walkable caves
      num_targets = max(2, len(self.drones) // 2)
      attempts = 0
      while len(self.targets) < num_targets and attempts < 100:
          tx = random.randint(0, self.map_w - 1)
          ty = random.randint(0, self.map_h - 1)
          
          # Check if location is floor (not wall)
          if ty < len(self.map_matrix) and tx < len(self.map_matrix[0]):
              if self.map_matrix[ty][tx] == 0:  # Floor cell
                  # Ensure target not too close to start positions
                  too_close = False
                  for start_pos in [d.start_pos for d in self.drones]:
                      if math.dist((tx, ty), start_pos) < 100:
                          too_close = True
                          break
                  
                  if not too_close:
                      self.targets.append(TargetEntity(
                          id=f"target_{len(self.targets)}",
                          location=(tx, ty)
                      ))
          
          attempts += 1
      
      print(f"Generated {len(self.targets)} rescue targets")
  ```

### 5.3 Modify Drone.scan_terrain() for Target Discovery

**File**: Drone.py — `scan_terrain()` method, end

- [ ] **Add target discovery check** (at very end of method):
  ```python
  # NEW: Check for nearby rescue targets (Search & Rescue mission)
  if hasattr(self.control, 'mission') and self.control.mission == 1:
      for target in self.control.targets:
          if not target.rescued and math.dist(self.pos, target.location) < 50:
              if self.id not in target.discovered_by:
                  target.discovered_by.append(self.id)
                  target.discovered_time = time.time()
                  print(f"[DISCOVERY] Drone {self.id} discovered target {target.id} at {target.location}")
  ```

### 5.4 Modify Drone.mission_completed() for Rescue Logic

**File**: Drone.py — `mission_completed()` method

- [ ] **Update to handle S&R** (modify existing method):
  ```python
  def mission_completed(self) -> bool:
      """Check if drone has completed its mission."""
      
      # Battery depletion check (from Phase 3)
      if self.battery <= 0:
          if not self.done:
              self.done = True
              self.returning_home = False
          return True
      
      # Not explored yet
      if not self.explored:
          return False
      
      # SEARCH & RESCUE mission: requires all targets rescued
      if hasattr(self.control, 'mission') and self.control.mission == 1:
          if not self.control.targets:
              # No targets generated - mission complete
              return True
          
          all_rescued = all(target.rescued for target in self.control.targets)
          if all_rescued and self.done:
              print(f"Drone {self.id}: All targets rescued - MISSION COMPLETE")
              return True
          
          if not all_rescued:
              return False  # Targets still pending
      
      # EXPLORATION mission: original behavior
      if not self.border and not self.done:
          self.returning_home = True
          return False
      
      # Recharge at base (from Phase 3)
      if self.done and math.dist(self.pos, self.start_pos) < 20:
          self.battery = 100
          return True
      
      return self.done
  ```

### 5.5 Add Target Rescue Mechanism in MissionControl

**File**: MissionControl.py — Main mission loop (drone_thread)

- [ ] **Add rescue check** (in drone_thread, after mission update loop):
  ```python
  # NEW: Check for rescues (drones at target location)
  if self.mission == 1:  # S&R mode
      for target in self.targets:
          if not target.rescued:
              for drone in self.drones:
                  if drone.id in target.discovered_by and math.dist(drone.pos, target.location) < 30:
                      target.rescued = True
                      target.rescued_by = drone.id
                      target.rescue_time = time.time()
                      print(f"[RESCUE] Drone {drone.id} rescued target {target.id}")
                      break
  ```

### 5.6 Add Rescue Status to ControlCenter Display

**File**: ControlCenter.py — Status display section

- [ ] **Add target status display** (add new section for mission status):
  ```python
  # In the draw_status() or status display loop, add:
  if hasattr(self.control, 'mission') and self.control.mission == 1:
      target_info = f"Targets: {sum(1 for t in self.control.targets if t.rescued)}/{len(self.control.targets)}"
      self.draw_text(target_info, 10, 100, Colors.WHITE)
      
      for target in self.control.targets:
          status = "RESCUED" if target.rescued else "PENDING"
          color = Colors.GREEN if target.rescued else Colors.RED
          target_text = f"  {target.id}: {status}"
          # Draw near target location on map
  ```

### 5.7 Validation Checklist

- [ ] Exploration missions: missions run unchanged (no targets)
- [ ] S&R missions: 2-3 targets randomly generated in walkable caves
- [ ] Drones discover targets when within 50px
- [ ] Discoveries broadcast to nearby drones via sharing (Phase 1)
- [ ] UI shows target count and rescue status
- [ ] Mission completes only when all targets rescued
- [ ] Print statements confirm discovery and rescue events

---

## PHASE 6: Drift Modeling (Polish Phase - Optional)

### 6.1 Add Position Tracking to Drone.__init__

**File**: Drone.py — `__init__` method, near end

- [ ] **Add position state** (after core position init):
  ```python
  # Drift modeling: ground truth vs estimated position
  self.actual_pos: Tuple[float, float] = start_pos  # System ground truth
  self.estimated_pos: Tuple[float, float] = start_pos  # Drone's noisy estimate
  self.accumulated_error: float = 0.0  # Total position uncertainty (pixels)
  self.drift_per_move: float = 0.5  # Configurable noise std dev
  self.last_correction_pos: Tuple[float, float] = start_pos
  self.last_correction_time: float = 0.0
  ```

### 6.2 Modify Drone.move() for Drift Accumulation

**File**: Drone.py — `move()` method, end

- [ ] **Add drift logic** (at very end, after existing battery/return-home logic):
  ```python
  # NEW: Update actual position (ground truth maintained by system)
  self.actual_pos = self.pos
  
  # NEW: Accumulate drift on estimated position using Gaussian noise
  import random as rand
  drift_x = rand.gauss(0, self.drift_per_move)
  drift_y = rand.gauss(0, self.drift_per_move)
  
  self.estimated_pos = (
      self.estimated_pos[0] + drift_x,
      self.estimated_pos[1] + drift_y
  )
  
  # Track accumulated error
  drift_magnitude = math.sqrt(drift_x**2 + drift_y**2)
  self.accumulated_error += drift_magnitude
  
  # NEW: Rendezvous correction - reset drift when near rover
  if hasattr(self, 'control') and self.control.rovers:
      for rover in self.control.rovers:
          rover_proximity = math.dist(self.actual_pos, rover.pos)
          if rover_proximity < 30:  # Within 30px of rover
              self.estimated_pos = self.actual_pos
              self.accumulated_error = 0.0
              self.last_correction_pos = self.actual_pos
              self.last_correction_time = time.time()
              print(f"Drone {self.id} corrected position at rover rendezvous")
              break
  ```

### 6.3 Modify Drawing Code to Use Estimated Position

**File**: Drone.py — Find all drawing/rendering code

- [ ] **Search for `self.pos` in rendering context** and replace with `self.estimated_pos` where visual:
  ```python
  # Example: icon drawing
  def draw_icon(self, window):
      """Render drone as estimated position with uncertainty halo."""
      # Draw uncertainty circle (visual indication of drift)
      pygame.draw.circle(
          window,
          (*self.color, 50),  # Semi-transparent color
          (int(self.estimated_pos[0]), int(self.estimated_pos[1])),
          int(max(1, self.accumulated_error)),
          width=1
      )
      
      # Draw drone icon at estimated position
      window.blit(self.icon, 
                 (int(self.estimated_pos[0]) - 8, int(self.estimated_pos[1]) - 8))
      
      # Optional: Draw faint ground truth for debugging
      if self.control.debug_mode:
          pygame.draw.circle(window, (128, 128, 128), 
                            (int(self.actual_pos[0]), int(self.actual_pos[1])), 2)
  ```

- [ ] **Keep collision/pathfinding on actual_pos**:
  ```python
  # Pathfinding uses actual_pos (ground truth)
  goal = self.reach_border(self.actual_pos)  # NOT estimated_pos
  
  # Graph checks use actual_pos
  if self.graph.is_valid(self.actual_pos):  # NOT estimated_pos
      pass
  ```

### 6.4 Add Drift Status to ControlCenter

**File**: ControlCenter.py — Status display

- [ ] **Add drift info** (optional display in status area):
  ```python
  # Add to drone status display loop:
  if self.control.debug_mode:
      drift_text = f"Drift: {drone.accumulated_error:.1f}px"
      self.draw_text(drift_text, drone_x, drone_y + 40, Colors.CYAN)
  ```

### 6.5 Validation Checklist

- [ ] Estimated position diverges from actual position during movement
- [ ] Drift corrects to zero when drone reaches rover
- [ ] Uncertainty halo grows then shrinks (visual feedback)
- [ ] Collisions use actual_pos (no ghost collisions)
- [ ] Pathfinding reaches actual goals (drones don't miss targets)
- [ ] Thermal image or heatmap uses actual_pos (not estimated)
- [ ] Run 500+ moves; verify drift < 50px by rendezvous

---

## Cross-Phase Integration Tests

### Post-Implementation Validation

- [ ] **All phases compile**: No import errors after integrating all phases
- [ ] **Both mission modes**: Run Exploration and S&R with all features enabled
- [ ] **Concurrency**: Run with 4+ drones; verify no race conditions or crashes
- [ ] **Progression**: Exploration → shared data → waypoints → targets rescue → mission end
- [ ] **UI responsiveness**: Battery, POI, targets, drift all display correctly
- [ ] **Baseline test**: Run with seed=12345; verify exploration patterns still valid

### Regression Test Checklist

- [ ] Existing exploration logic unchanged for Exploration missions
- [ ] Drone-rover terrain sharing still works (S&R + Exploration)
- [ ] AStar pathfinding accuracy not degraded
- [ ] Simulation frame rate acceptable (<50ms per frame)
