import pathlib
import sys
import joblib

BASE_DIR = pathlib.Path(__file__).resolve().parents[1]
BACKEND_DIR = BASE_DIR / "cyber_detect_backend-master"
sys.path.append(str(BACKEND_DIR))

from randomforesttest import train_randomforest


def main():
    model = train_randomforest()
    out_path = BACKEND_DIR / "random_forest_model.pkl"
    joblib.dump(model, out_path)
    print(f"Saved RandomForest model to {out_path}")


if __name__ == "__main__":
    main()

