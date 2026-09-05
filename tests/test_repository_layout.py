from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"


class RepositoryLayoutTests(unittest.TestCase):
    def test_root_is_flat_and_does_not_restore_retired_wrappers(self):
        self.assertTrue(SRC.is_dir())
        for retired in (
            ROOT / "main.py",
            ROOT / "sam3_auto_annotator",
            ROOT / "images_test",
            SRC / "sam3_auto_annotator",
            SRC / "gui" / "controller.py",
        ):
            with self.subTest(path=retired.relative_to(ROOT)):
                self.assertFalse(retired.exists())

    def test_expected_source_boundaries_live_directly_under_src(self):
        for path in (
            SRC / "main.py",
            SRC / "domain",
            SRC / "gui",
            SRC / "sam3",
            SRC / "services",
            SRC / "storage",
        ):
            with self.subTest(path=path.relative_to(ROOT)):
                self.assertTrue(path.exists())

    def test_entrypoint_is_real_composition_root_not_a_forwarder(self):
        source = (SRC / "main.py").read_text(encoding="utf-8")
        self.assertIn("QApplication", source)
        self.assertIn("WorkstationController", source)
        self.assertIn('if __name__ == "__main__":', source)
        self.assertNotIn("from application import", source)
        self.assertNotIn("import application", source)
        self.assertNotIn("sam3_auto_annotator", source)

    def test_removed_namespace_is_not_imported_anywhere(self):
        offenders = []
        for base in (SRC, ROOT / "tests"):
            for path in base.rglob("*.py"):
                body = path.read_text(encoding="utf-8")
                if (
                    "from sam3_auto_annotator" in body
                    or "import sam3_auto_annotator" in body
                    or "from gui.controller" in body
                    or "import gui.controller" in body
                ):
                    offenders.append(str(path.relative_to(ROOT)))
        self.assertEqual(offenders, [])

    def test_one_shot_migration_workflows_are_not_part_of_repository(self):
        workflow_dir = ROOT / ".github" / "workflows"
        retired = {
            "layout-migration.yml",
            "layout-migration-v2.yml",
            "fix-layout-tests.yml",
            "fix-layout-tests-v2.yml",
        }
        present = {path.name for path in workflow_dir.glob("*.yml")}
        self.assertTrue(retired.isdisjoint(present))


if __name__ == "__main__":
    unittest.main()
