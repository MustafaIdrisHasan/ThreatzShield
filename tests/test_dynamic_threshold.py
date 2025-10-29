import sys
import pathlib

BASE_DIR = pathlib.Path(__file__).resolve().parents[1]
BACKEND_DIR = BASE_DIR / "cyber_detect_backend-master"
sys.path.append(str(BACKEND_DIR))

import unittest
from ensemble import dynamic_threshold_prediction


class TestDynamicThreshold(unittest.TestCase):
    def test_dynamic_threshold_prediction_bounds(self):
        # If all confidences high, expect Normal
        label, score = dynamic_threshold_prediction(0.9, 0.9, 0.9)
        self.assertEqual(label, "Normal")
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)

        # If all confidences low, expect Cyberbullying
        label, score = dynamic_threshold_prediction(0.1, 0.1, 0.1)
        self.assertEqual(label, "Cyberbullying")
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)


if __name__ == "__main__":
    unittest.main()
