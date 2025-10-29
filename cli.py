import sys
import pathlib

BASE_DIR = pathlib.Path(__file__).parent
BACKEND_DIR = BASE_DIR / "cyber_detect_backend-master"
sys.path.append(str(BACKEND_DIR))

from ensemble import (
    predict_outputs,
    adjust_normal_percentage,
    dynamic_threshold_prediction,
)


def main():
    if len(sys.argv) < 2:
        print("Usage: python cli.py \"your text here\"")
        sys.exit(1)

    text = sys.argv[1]
    outputs = predict_outputs(text)
    bert_conf = outputs[1]
    bert_norm = adjust_normal_percentage(bert_conf[1] / 100, bert_conf[0] / 100, bert_conf[2] / 100)

    lstm_conf = outputs[0]
    rf_conf = outputs[2]
    rf_norm = adjust_normal_percentage(rf_conf[2], rf_conf[0], rf_conf[1])

    label, score = dynamic_threshold_prediction(bert_norm, lstm_conf[1], rf_norm)
    print({"label": label, "normal_score": score, "raw": outputs})


if __name__ == "__main__":
    main()

