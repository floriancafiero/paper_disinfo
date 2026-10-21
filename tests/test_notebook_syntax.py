"""Compile all revised standalone Python scripts and notebook code cells."""
import ast
import json
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = [
    ROOT / "colab_no_embeddings" / "evons_disinformation_detection.py",
    ROOT / "colab_no_embeddings" / "evons_virality_prediction.py",
    ROOT / "tools" / "colleague_revision_run.py",
]
NOTEBOOKS = [
    ROOT / "colab_no_embeddings" / f"{task}.ipynb"
    for task in (
        "evons_disinformation_detection", "evons_virality_prediction",
        "fakenewsnet_disinformation_detection", "fakenewsnet_virality_prediction",
    )
]


class NotebookSyntaxTests(unittest.TestCase):
    def test_python_scripts_compile(self):
        for file in SCRIPTS:
            with self.subTest(file=file.name):
                ast.parse(file.read_text(encoding="utf-8"), filename=str(file))

    # Check all code cells independently, including the nested CV baselines.
    def test_all_code_cells_compile(self):
        for file in NOTEBOOKS:
            notebook = json.loads(file.read_text(encoding="utf-8"))
            for i, cell in enumerate(notebook["cells"]):
                if cell["cell_type"] != "code":
                    continue
                with self.subTest(notebook=file.name, cell=i):
                    ast.parse("".join(cell["source"]), filename=f"{file.name} cell {i}")


if __name__ == "__main__":
    unittest.main()
