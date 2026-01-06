# Cave Game - AI Coding Assistant Instructions

## Project Overview
This is a distributed systems simulation game built with Pygame, demonstrating multi-agent cave exploration algorithms. Drones and rovers concurrently explore procedurally generated cave maps using vision-based frontier exploration.

## Architecture Overview

### Core Components
- **`Game`**: Main orchestrator handling pygame initialization, menu navigation, and simulation lifecycle
- **`MissionControl`**: Manages concurrent drone/rover operations using threading, coordinates exploration missions
- **`MapGenerator`**: Parallel map generation using multiprocessing "worms" that erode solid terrain
- **`Drone`/`Rover`**: Autonomous agents with vision cones, random walk exploration, and A* pathfinding
- **`AStar`**: Pathfinding for reaching unexplored frontiers
- **`Graph`**: Maintains exploration connectivity graphs
- **`ControlCenter`**: Real-time mission status display
- **`Assets`**: Centralized enum-based asset and constant management

### Key Design Patterns

#### Threading for Concurrency
```python
# MissionControl uses threading for concurrent drone movement
threads = []
for i in range(self.num_drones):
    t = threading.Thread(target=self.drone_thread, args=(i,))
    threads.append(t)
    t.start()
```

#### Multiprocessing for Map Generation
```python
# MapGenerator spawns parallel processes for cave erosion
for i in range(proc_num):
    proc_list.append(Process(target=self.worm(...)))
    proc_list[i].start()
```

#### Vision-Based Exploration
```python
# Drones cast rays to detect walls and build vision polygons
def cast_ray(self, start_pos, angle, max_length):
    # Ray casting for obstacle detection
```

#### Layered Drawing System
```python
# Draw in layers: vision -> paths -> walls -> icons
def draw(self):
    self.draw_cave()
    for layer in [vision, path, walls, icons]:
        # Layer-specific rendering
```

## Development Workflow

### Running the Simulation
```bash
python main.py
```
- Navigate menus with arrow keys and Enter
- Configure mission type, map size, drone count, random seed
- Simulation runs until all drones complete exploration

### Debugging
- Use VS Code Python debugger with `launch.json`
- Mission runs in threads; use breakpoints in `drone_thread()` for drone-specific debugging
- Map generation happens in separate processes; debug worms individually

### Configuration
- `GameConfig/options.ini`: Audio settings
- `GameConfig/symSettings.ini`: Default simulation parameters
- Random seed ensures reproducible map generation

## Code Conventions

### Asset Management
```python
# Use enums for all assets and constants
class Images(Enum):
    CAVE_MAP = os.path.join(GAME_DIR, 'Assets', 'Map', 'map.png')

# Access via Assets.Images['CAVE_MAP'].value
```

### Surface-Based Rendering
```python
# Create transparent surfaces for overlays
self.floor_surf = pygame.Surface((width, height), pygame.SRCALPHA)
# Draw to surface, then blit to main window
self.game.window.blit(self.floor_surf, (0,0))
```

### Error Handling
```python
try:
    pygame.init()
except pygame.error as e:
    print(f"Failed to initialize Pygame: {e}")
    sys.exit(1)
```

### Type Hints and Modern Python
```python
def __init__(self) -> None:
def run(self) -> NoReturn:
```

## Dependencies
- `pygame`: Graphics and input handling
- `numpy`: Matrix operations for map generation
- `cv2` (opencv): Image processing for cave refinement
- `art`: ASCII art (used in Test.py)

## Common Tasks

### Adding New Agent Behaviors
1. Extend `Drone` or `Rover` class with new strategy
2. Update `mission_completed()` logic
3. Add visualization in `draw_*()` methods

### Modifying Map Generation
1. Adjust `WormInputs` enum values for different cave styles
2. Modify `worm()` method for new erosion patterns
3. Update `process_map()` for post-processing effects

### Performance Optimization
- Reduce `num_rays` in vision casting for faster rendering
- Adjust `delay` in MissionControl for simulation speed
- Profile multiprocessing overhead vs single-threaded generation

## File Organization
- `main.py`: Entry point
- `Game.py`: Core game loop and menu handling
- `MissionControl.py`: Simulation orchestration
- `Drone.py`/`Rover.py`: Agent implementations
- `MapGenerator.py`: Procedural cave generation
- `Assets.py`: Centralized constants and enums
- `AStar.py`/`Graph.py`: Pathfinding and graph management
- `Menu.py`: UI components
- `ControlCenter.py`: Status display