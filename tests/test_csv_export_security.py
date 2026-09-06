import csv
import tempfile
import unittest
from pathlib import Path

from storage.exporters.csv_exporter import spreadsheet_safe_cell, write_csv


class CsvExportSecurityTests(unittest.TestCase):
    def test_dangerous_text_prefixes_are_neutralized(self):
        for value in (
            "=1+1",
            "+SUM(A1:A2)",
            "-2+3",
            "@SUM(A1:A2)",
            "\t=1+1",
            "\r=1+1",
            "\n=1+1",
        ):
            with self.subTest(value=value):
                self.assertEqual(spreadsheet_safe_cell(value), "'" + value)

    def test_numeric_and_normal_text_values_are_unchanged(self):
        for value in (0, -3.5, None, "car", "image-1.jpg", "0 0.5 0.5 0.2 0.2"):
            with self.subTest(value=value):
                self.assertEqual(spreadsheet_safe_cell(value), value)

    def test_write_csv_neutralizes_text_without_mutating_source_row(self):
        row = {
            "image_name": "=HYPERLINK(\"https://example.invalid\")",
            "class_name": "+cmd",
            "confidence": 0.75,
        }
        original = dict(row)
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "annotations.csv"
            write_csv(path, [row], ["image_name", "class_name", "confidence"])
            with path.open(newline="", encoding="utf-8") as csv_file:
                exported = next(csv.DictReader(csv_file))

        self.assertEqual(row, original)
        self.assertEqual(
            exported["image_name"],
            "'=HYPERLINK(\"https://example.invalid\")",
        )
        self.assertEqual(exported["class_name"], "'+cmd")
        self.assertEqual(exported["confidence"], "0.75")


if __name__ == "__main__":
    unittest.main()
