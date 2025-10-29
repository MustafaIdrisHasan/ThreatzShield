import sys
import pathlib

BASE_DIR = pathlib.Path(__file__).resolve().parents[1]
BACKEND_DIR = BASE_DIR / "cyber_detect_backend-master"
sys.path.append(str(BACKEND_DIR))

from ensemble import dynamic_threshold_prediction


def test_dynamic_threshold_prediction_bounds():
    # If all confidences high, expect Normal
    label, score = dynamic_threshold_prediction(0.9, 0.9, 0.9)
    assert label == "Normal"
    assert 0.0 <= score <= 1.0

    # If all confidences low, expect Cyberbullying
    label, score = dynamic_threshold_prediction(0.1, 0.1, 0.1)
    assert label == "Cyberbullying"
    assert 0.0 <= score <= 1.0

