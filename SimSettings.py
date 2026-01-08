from dataclasses import dataclass


@dataclass
class SimSettings:
    mission: int
    map_dim: str
    seed: int
    num_drones: int
    prefab: int
