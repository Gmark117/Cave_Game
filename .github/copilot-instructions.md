# Cave Game - AI Coding Assistant Instructions

## Project Overview
This is a Pygame-based distributed systems simulation game where drones explore procedurally generated cave maps. The project demonstrates multi-agent exploration algorithms, procedural map generation, and real-time visualization.

## Architecture Overview

### Core Components
- **Game.py**: Main game loop managing menu navigation and state transitions
- **MissionControl.py**: Simulation orchestrator handling threading, drone coordination, and mission completion
- **Drone.py**: Individual drone agents with exploration logic, vision systems, and pathfinding
- **MapGenerator.py**: Procedural cave generation using parallel "worm" algorithms
- **Menu Classes**: UI system (MainMenu, SimulationMenu, OptionsMenu, etc.)

### Key Design Patterns
- **State Machine**: Menu-driven navigation with distinct game states
- **Multi-threading**: Concurrent drone movement during simulation
- **Multiprocessing**: Parallel map generation for performance
- **Observer Pattern**: Event-driven input handling and state updates
- **Factory Pattern**: Dynamic creation of drones/rovers based on settings

**UI Architecture Note**: The current menu system uses inheritance-heavy design. Consider migrating to the component-based `UnifiedMenu` system for better maintainability.

## Development Workflow

### Running the Game
```bash
python main.py
```
- Navigate menus with arrow keys and Enter
- Configure simulation settings in SimulationMenu
- Watch real-time drone exploration in fullscreen mode

### Configuration Files
- `GameConfig/options.ini`: Audio and UI preferences
- `GameConfig/symSettings.ini`: Default simulation parameters
- Settings persist between sessions

### Map Generation
- Procedural generation using 8 parallel "worms" that erode the map
- Seed-based reproducibility for testing
- Support for prefab maps loaded from `Assets/Map/map_matrix.txt`

## Coding Conventions

### Import Organization
```python
import pygame
import Assets
from ComponentClass import ComponentClass
```
- Standard library first, then Assets, then local imports
- Use `from Assets import Colors, Images` for enum access

### Enum Usage
```python
# Colors
Assets.Colors.WHITE.value  # (255, 255, 255)

# Images  
Assets.Images.DRONE.value  # Path to drone image

# Constants
Assets.DISPLAY_W  # 1200
```

### Coordinate System
- Origin (0,0) at top-left
- X increases right, Y increases down
- Use `next_cell_coords(x, y, step, direction)` for movement calculations

### Threading Patterns
```python
# Drone threads in MissionControl
thread = threading.Thread(target=self.drone_thread, args=(drone_id,))
thread.start()
threads.append(thread)

# Wait for completion
for t in threads:
    t.join()
```

## Common Development Tasks

### Adding New Drone Behaviors
1. Modify `Drone.move()` method
2. Update exploration logic in `find_new_node()` and `explore()`
3. Test with different map seeds and sizes

### Modifying Map Generation
1. Adjust worm parameters in `Assets.WormInputs`
2. Update `MapGenerator.worm()` algorithm
3. Test generation speed vs. quality tradeoffs

### Adding UI Elements
1. Extend menu classes inheriting from `Menu`
2. Add states to `Assets.sim_menu_states`
3. Implement `display()` and `check_input()` methods

**Note**: Consider migrating to the new `UnifiedMenu` system (see `UI_REFACTOR_PROPOSAL.md` and `UnifiedMenu.py`) for simplified UI management.

### UI Refactoring (Recommended)
The current menu system has significant code duplication. A streamlined approach using `UnifiedMenu.py`:
- Define menus as data structures instead of classes
- Unified input handling and rendering
- Component-based architecture with `MenuItem` subclasses
- Reduces code by ~70% and improves maintainability

### Performance Optimization
- Map generation uses multiprocessing for CPU-intensive tasks
- Simulation uses threading for concurrent drone updates
- Vision calculations use ray casting for realistic field-of-view

## Dependencies
- pygame: Graphics and input handling
- numpy: Matrix operations for map data
- opencv-python: Image processing for map generation
- art: ASCII art (optional, used only in Test.py)

## Testing Approach
- Visual testing: Run simulations with different seeds
- Unit testing: Validate pathfinding algorithms in isolation
- Integration testing: Full simulation runs with multiple drones

## Key Files for Understanding
- `MissionControl.py`: Simulation orchestration and threading
- `Drone.py`: Core exploration algorithms and vision system
- `MapGenerator.py`: Procedural generation with multiprocessing
- `Assets.py`: All constants, enums, and utility functions
- `SimulationMenu.py`: Configuration interface and settings management