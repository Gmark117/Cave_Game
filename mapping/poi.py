"""Point of Interest (POI) dataclass for autonomous drone discovery.

POIs are discovered by drones through topology analysis:
- Chambers: local minima (smooth areas surrounded by rougher terrain)
- Formations: wall obstacles (black pixels) with curved approach patterns
"""

from dataclasses import dataclass, field
from typing import Set, List, Tuple, Optional


@dataclass
class POI:
    """Represents a discovered or candidate Point of Interest in the cave.
    
    Attributes:
        id: Unique identifier for the POI
        poi_type: "chamber" or "formation"
        location: (x, y) center coordinates of the POI
        required_cells: Set of cells that define the POI boundary/area
        discovered_by: List of drone IDs that have verified this POI
        discovered_time: Unix timestamp when POI was verified (None if candidate)
        start_pos: Origin position when candidate was detected (formation candidates)
        initial_blind_angle: Vision blind spot angle at detection (formation candidates)
    """
    id: str
    poi_type: str  # "chamber" or "formation"
    location: Tuple[int, int]  # (x, y) center
    required_cells: Set[Tuple[int, int]] = field(default_factory=set)
    discovered_by: List[int] = field(default_factory=list)
    discovered_time: Optional[float] = None
    start_pos: Optional[Tuple[int, int]] = None
    initial_blind_angle: Optional[float] = None

    def __hash__(self):
        """Make POI hashable by ID."""
        return hash(self.id)

    def __eq__(self, other):
        """POIs are equal if they have the same ID."""
        if not isinstance(other, POI):
            return False
        return self.id == other.id
