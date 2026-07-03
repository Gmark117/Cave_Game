import os
import tempfile
import unittest
from pathlib import Path

import numpy as np

os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

from generation.map_artifact_writer import MapArtifactWriter


class MapArtifactWriterTests(unittest.TestCase):
    def test_save_map_writes_image_and_matrix_under_project_assets(self) -> None:
        bin_map = np.array([[1, 0], [0, 1]], dtype=np.uint8)

        with tempfile.TemporaryDirectory() as temp_dir:
            writer = MapArtifactWriter(Path(temp_dir))
            writer.save_map(bin_map)

            map_dir = Path(temp_dir) / "Assets" / "Map"
            self.assertTrue((map_dir / "map.png").exists())
            self.assertTrue((map_dir / "map_matrix.txt").exists())
            saved_matrix = np.loadtxt(map_dir / "map_matrix.txt")
            np.testing.assert_array_equal(saved_matrix, bin_map)


if __name__ == "__main__":
    unittest.main()
