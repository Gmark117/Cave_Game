# Cave Game

Distributed Systems project that simulates concurrent cave exploration with autonomous drones and rovers.

## Overview

Cave Game demonstrates multi-agent coordination in a procedurally generated environment. Agents explore a cave map concurrently, share mission-relevant information, and progressively build global situational awareness.

Main building blocks:

- Threaded mission orchestration for multiple agents
- Procedural cave generation
- Vision-based local exploration
- Frontier/path-driven movement behaviors

## Prerequisites

- Python 3.11+
- Windows, Linux, or macOS
- `pip` available in your Python installation

## Quick Start

Create and activate a virtual environment (recommended):

```bash
python -m venv .venv
```

Windows PowerShell:

```bash
.venv\Scripts\Activate.ps1
```

Windows CMD:

```bash
.venv\Scripts\activate.bat
```

Install dependencies:

```bash
python -m pip install --upgrade pip
python -m pip install pygame numpy opencv-python
```

Run the game:

```bash
python main.py
```

## Controls

In menus:

- `Up` / `Down`: move selection
- `Left` / `Right`: change selector or slider values
- `Enter`: confirm / open submenu / start mission
- Number keys `0-9`: edit seed field
- `Backspace`: delete seed digits

Simulation settings available in-game:

- Objective: `Exploration` or `Search and Rescue`
- Cave size: `Small`, `Medium`, `Big`
- Seed: custom numeric seed or default per cave size
- Drones: from 3 to 8
- Demo Cave: `Yes`/`No`

## Configuration

Persistent settings are stored in:

- `GameConfig/options.ini`: audio volume, music toggle, button sound toggle
- `GameConfig/symSettings.ini`: last selected mission parameters

## Project Status

| Feature | Status | Notes |
|---|---|---|
| Terrain roughness map | Implemented | Available in current simulation flow |
| Known map visualization | Implemented | Available in current simulation flow |
| Distributed data sharing (terrain, POI, paths) | In progress | Core sharing exists, integration is ongoing |
| Waypoints for optimal path segmentation | Planned | Not yet implemented |
| Battery management | Planned | Not yet implemented |
| Non-random exploration logic | Planned | Current behavior includes random exploration components |
| Search & Rescue mission logic | Planned | Objective exists in UI; full mission logic is pending |
| Drift modeling | Planned | Not yet implemented |

## Documentation

Additional documentation is available in the `docs` folder:

- `docs/DISTRIBUTED_MAPPING.md`
- `docs/IMPLEMENTATION_SUMMARY.md`
- `docs/INTEGRATION_GUIDE.md`
- `docs/MapGenerator.md`

## Troubleshooting

- If dependencies fail to install, upgrade `pip` first and retry.
- If `pygame` audio initialization fails, check that your system audio device is available and not locked by another app.
- If `cv2` import fails, reinstall OpenCV:

```bash
python -m pip install --force-reinstall opencv-python
```

## Notes

- This repository is under active development.
- Some modules contain extension points for future mission logic.
