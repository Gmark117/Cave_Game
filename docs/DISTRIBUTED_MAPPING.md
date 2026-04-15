# Distributed Terrain Mapping System

## Overview

The simulation now implements a **distributed terrain knowledge model** where each drone maintains its own local terrain map. Data is shared only when drones encounter each other or when they interact with rovers, simulating realistic distributed systems constraints.

## Architecture

### Per-Drone Local Knowledge

Each `Drone` instance maintains its own terrain maps:
- `known_roughness`: Float32 array storing estimated terrain roughness values [0,1]
- `terrain_confidence`: Float32 array storing observation confidence [0,1]
- `terrain_lock`: Thread-safe lock protecting map updates

**Initialization** (in `Drone.__init__`):
```python
self.known_roughness = np.full(np.asarray(self.cave).shape, -1.0, dtype=np.float32)
self.terrain_confidence = np.zeros(np.asarray(self.cave).shape, dtype=np.float32)
self.terrain_lock = threading.Lock()
```

### Terrain Scanning (Local Recording)

When a drone scans, it updates **only its own maps** via `_record_local_terrain_scan()`:
- Samples terrain roughness along 96 rays from the drone's position
- Records samples with distance-based confidence (closer = higher confidence)
- Uses weighted averaging when merging new observations with existing knowledge

```python
def scan_terrain(self) -> None:
    # ... ray casting logic ...
    self._record_local_terrain_scan(samples)  # Updates self.known_roughness/confidence
```

### Data Sharing Mechanisms

#### 1. **Proximity-Based Drone-to-Drone Sharing**

When two drones come within **200 pixels** of each other, they automatically exchange terrain data:

```python
def _share_terrain_with_nearby_drones(self, drone_id: int) -> None:
    """Called every 0.5s from drone_thread()"""
    for other_drone in self.drones:
        distance = math.sqrt((drone.pos[0] - other.pos[0])**2 + ...)
        if distance < 200:  # Proximity threshold
            # Bidirectional merge
            drone.merge_terrain_data(other.known_roughness, other.terrain_confidence)
            other.merge_terrain_data(drone.known_roughness, drone.terrain_confidence)
```

**Key Points:**
- Sharing is **bidirectional**: both drones update with each other's data
- Data exchange uses **weighted averaging**: higher-confidence observations are weighted more
- Sharing is **throttled** to 0.5s intervals to prevent excessive lock contention
- Thread-safe via `terrain_lock` on each drone

#### 2. **Rover as Data Hub**

Rovers accumulate terrain knowledge from nearby drones (within **150 pixels**):

```python
def _share_terrain_with_rovers(self) -> None:
    """Called once per main loop frame"""
    for rover in self.rovers:
        for drone in self.drones:
            if distance(rover, drone) < 150:
                rover.known_roughness = merge(rover.known_roughness, drone.known_roughness)
                rover.terrain_confidence = merge(rover.terrain_confidence, drone.terrain_confidence)
```

**Rover Properties:**
- Rovers are stationary at a central location (starting point)
- Act as **persistent data collectors** that accumulate observations
- Drones automatically share whenever they pass near the rover
- Rovers can serve as rendezvous points for data synchronization

### Terrain Merging Algorithm

Both `merge_terrain_data()` (drone-to-drone) and the rover aggregation use the same weighted average:

```python
For each cell (x, y):
    if cell is floor (traversable):
        other_conf = other_drone.terrain_confidence[y, x]
        if other_conf <= 0:
            continue  # No data from other drone
        
        other_roughness = other_drone.known_roughness[y, x]
        self_conf = self.terrain_confidence[y, x]
        self_roughness = self.known_roughness[y, x] or other_roughness
        
        total_confidence = self_conf + other_conf
        merged = (self_roughness * self_conf + other_roughness * other_conf) / total_confidence
        
        self.known_roughness[y, x] = merged
        self.terrain_confidence[y, x] = min(1.0, total_confidence)
```

**Properties:**
- Observations with higher confidence are weighted more heavily
- Confidence accumulates (compounds with each new observation)
- Capped at 1.0 to prevent unbounded growth
- Requires both drones to have valid data before merging

## Heatmap Visualization

The terrain heatmap now displays **aggregated knowledge** from all agents:

1. **Data Aggregation** (`_refresh_terrain_heatmap()`):
   - Merges terrain maps from all drones
   - Includes data from rovers (if they've collected any)
   - Creates a composite visualization of the team's collective knowledge

2. **Color Coding** (5-band discrete palette):
   - **Blue**: Very smooth (low roughness)
   - **Green**: Smooth
   - **Yellow**: Medium roughness
   - **Orange**: Rough
   - **Red**: Very rough (high roughness)

3. **Opacity** (confidence-based):
   - Alpha = 35 + (confidence × 125)
   - Cells with more observations appear more opaque
   - Unexplored areas remain transparent

## Simulation Parameters

**Sharing Intervals:**
- Drone-to-drone sharing check: **0.5s** (throttled)
- Rover sharing: **Every frame** (main loop)
- Heatmap refresh: **0.25s** (throttled)

**Proximity Thresholds:**
- Drone-to-drone sharing: **200 pixels**
- Drone-to-rover sharing: **150 pixels**

**Scan Parameters:**
- Ray count per scan: **96 rays**
- Scan throttling interval: **0.25s**
- Confidence decay with distance: `max(0.2, 1.0 - (length / radius))`

## Key Differences from Previous Model

| Aspect | Before | After |
|--------|--------|-------|
| **Terrain maps** | Global (MissionControl) | Per-drone local |
| **Data sharing** | Immediate (centralized) | Event-driven (proximity) |
| **Rover role** | No terrain awareness | Data accumulator |
| **Heatmap source** | Single shared map | Aggregated from all drones |
| **Scalability** | Couples all agents | Scales to many agents |
| **Realism** | Unrealistic omniscience | Distributed knowledge |

## Thread Safety

All terrain map updates are protected by per-drone locks:

```python
with drone.terrain_lock:
    drone.known_roughness[y, x] = new_value
    drone.terrain_confidence[y, x] = new_conf
```

Sharing operations copy data before merging to minimize lock hold time:
```python
with other_drone.terrain_lock:
    other_copy = other_drone.known_roughness.copy()
    other_conf = other_drone.terrain_confidence.copy()

# Merge outside the lock
with self.terrain_lock:
    self.merge_terrain_data(other_copy, other_conf)
```

## Usage Example

When running the simulation:
1. Drones scan and build individual terrain maps as they explore
2. When two drones pass within 200 pixels, they automatically exchange data
3. Each drone benefits from observations made by other drones
4. The rover acts as a neutral meeting ground, accumulating knowledge
5. The heatmap visualizes the entire team's collective understanding

This creates emergent behavior where the team's map knowledge grows more complete as drones interact and collaborate, realistic of actual distributed robotic systems.
