"""Quantile grid configuration tests, no corpus or GPU needed."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from run_virality_quantile_grid import validate_quantiles


class QuantileGridTests(unittest.TestCase):
    def test_default_grid(self):
        self.assertEqual(validate_quantiles("0.5,0.75,0.9,0.95"),
                         [0.5, 0.75, 0.9, 0.95])
    def test_reject_bad_grid(self):
        for value in ("0", "1", "0.5,1.2", "0.5,0.5", ""):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    validate_quantiles(value)


if __name__ == "__main__":
    unittest.main()
