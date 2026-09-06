import tempfile
import unittest
from pathlib import Path

from PIL import Image

from services.project_service import create_project, verify_source_image_sizes
from storage.project_store import load_project_state, save_project_state


class SourceFingerprintIntegrityTests(unittest.TestCase):
    def test_same_dimension_source_replacement_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            image_path = Path(directory) / "frame.png"
            Image.new("RGB", (20, 10), "red").save(image_path)
            project = create_project(image_path, ["car"])
            original_digest = project.images[0].source_sha256

            Image.new("RGB", (20, 10), "blue").save(image_path)
            self.assertEqual(project.images[0].width, 20)
            self.assertEqual(project.images[0].height, 10)
            with self.assertRaisesRegex(ValueError, "content changed"):
                verify_source_image_sizes(project)
            self.assertEqual(project.images[0].source_sha256, original_digest)

    def test_fingerprint_round_trips_through_project_state(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image_path = root / "frame.png"
            state_path = root / "annotation_state.json"
            Image.new("RGB", (8, 6), "white").save(image_path)
            project = create_project(image_path, ["car"])
            save_project_state(project, state_path)

            loaded = load_project_state(state_path)
            image = loaded.images[0]
            self.assertEqual(len(image.source_sha256), 64)
            self.assertEqual(image.source_size_bytes, image_path.stat().st_size)
            self.assertEqual(image.source_mtime_ns, image_path.stat().st_mtime_ns)


if __name__ == "__main__":
    unittest.main()
