ThreatzShield – Cyberbullying Detection (BERT + LSTM + RandomForest)

Overview
- Detects hate/offensive language vs. normal text using a heterogeneous ensemble:
  - BERT (HateXplain) via HuggingFace Transformers.
  - LSTM (Keras/TensorFlow) pre-trained model.
  - RandomForest (scikit-learn) trained on a cleaned bag-of-words representation.
- Robust fallbacks ensure the project runs in constrained/offline setups (see Fallbacks).

Architecture
- Preprocessing: `randomforesttest.clean(text)` lowercases, removes punctuation/digits/URLs/HTML, drops stopwords, applies Snowball stemming.
- Models:
  - BERT (`cyber_detect_backend-master/berttest2.py`): returns class probabilities for [hate, normal, offensive] in percentages.
  - LSTM (`cyber_detect_backend-master/model3.h5`): binary distribution [hate, normal]; the ensemble uses only the Normal component.
  - RandomForest (`cyber_detect_backend-master/randomforesttest.py`): CountVectorizer → RandomForestClassifier, predicts 3-class probabilities.
- Ensemble (`cyber_detect_backend-master/ensemble.py`):
  - Adjusts Normal probability with `adjust_normal_percentage()` to avoid misweighting when Normal is the dominant class.
  - Combines confidences with weights: BERT 0.6, LSTM 0.3, RF 0.1; threshold 0.5 for final label.
  - Exposes `predict_outputs(text)` and `dynamic_threshold_prediction(...)` and is safe to import (no side effects).

Tech Stack
- Python 3.12
- ML: TensorFlow/Keras, PyTorch + Transformers, scikit-learn, pandas, numpy, nltk
- Serving: FastAPI + Uvicorn
- Utilities: joblib, tqdm, PyYAML, Pillow
- Testing: unittest
- Tooling: Makefile targets, `requirements.txt` (curated), `requirements-freeze.txt` (exact venv freeze)

Repository Layout
- Backend code: `cyber_detect_backend-master/`
- Ensemble logic: `cyber_detect_backend-master/ensemble.py`
- BERT wrapper: `cyber_detect_backend-master/berttest2.py`
- RF training/inference: `cyber_detect_backend-master/randomforesttest.py`
- Pretrained artifacts: `cyber_detect_backend-master/model3.h5`, `cyber_detect_backend-master/random_forest_model.pkl`
- Data (example): `cyber_detect_backend-master/twitter_data.csv`
- CLI: `cli.py` – quick local predictions
- API: `api.py` – FastAPI service (`POST /predict`)
- Tests: `tests/` – `test_preprocess.py`, `test_dynamic_threshold.py`
- Env pin: `requirements-freeze.txt`

Setup
- Option A: Fresh environment (recommended)
  - `python -m venv .venv && .venv/Scripts/activate` (Windows) or `source .venv/bin/activate` (Linux/Mac)
  - `pip install -r cyber_detect_backend-master/requirements.txt`
- Option B: Use the bundled freeze
  - `pip install -r requirements-freeze.txt` (may be heavy and platform-specific)
- Windows-only convenience (if you keep a local venv under the backend):
  - Python path: `cyber_detect_backend-master/.venv/Scripts/python.exe`
  - Uvicorn path: `cyber_detect_backend-master/.venv/Scripts/uvicorn.exe`

Run (CLI)
- Example: `python cli.py "Your text here"`
- Windows venv example: `cyber_detect_backend-master/.venv/Scripts/python.exe cli.py "Your text here"`

Run (API)
- Start: `uvicorn api:app --reload`
- Windows venv example: `cyber_detect_backend-master/.venv/Scripts/uvicorn.exe api:app --reload`
- Predict:
  - `curl -X POST http://127.0.0.1:8000/predict -H "Content-Type: application/json" -d '{"text":"Hello world"}'`

Testing
- Unit tests (fast, offline):
  - `python -m unittest discover -s tests -p "test_*.py"`
- Functionality probe (slower, touches real models):
  - `python cyber_detect_backend-master/test_functionality.py`

Fallbacks and Robustness
- BERT (`berttest2.py`):
  - Attempts local model first: set `HATE_MODEL_DIR` or place files under `cyber_detect_backend-master/models/hatexplain/`.
  - Falls back to HuggingFace hub download; if unavailable (offline), returns a uniform distribution to keep the pipeline running.
- LSTM (`ensemble.py`):
  - If TensorFlow or the H5 model fails to load, uses a neutral `[0.5, 0.5]` distribution.
- RandomForest (`randomforesttest.py`):
  - If `random_forest_model.pkl` is missing or incompatible, trains a new model on `twitter_data.csv` automatically.
  - Data file is loaded via a file-relative path for stability.
- NLTK stopwords are optional; a small built-in list is used if NLTK resources are unavailable.

What Was Improved In This Iteration
- Added a clean CLI (`cli.py`) and a small FastAPI service (`api.py`).
- Refactored `ensemble.py` to avoid side effects on import and to return (label, score).
- Hardened BERT and NLTK fallbacks for offline use.
- Ensured RF data loading is file-relative; auto-trains RF when the pickle is incompatible.
- Added minimal unit tests and a Makefile for quick commands.
- Generated `requirements-freeze.txt` to reproduce the current environment.

Known Limitations / Next Steps
- LSTM model compatibility: the shipped `model3.h5` may be incompatible with the current Keras (e.g., unexpected `time_major`).
  - Fix: re-export/retrain with the active TensorFlow/Keras version.
- RandomForest pickle compatibility can break across scikit-learn versions.
  - Fix: re-save the pickle with the target scikit-learn, or always retrain on start.
- Performance and metrics are not formalized in this repo (no F1/PR AUC table yet).
  - Add an evaluation script and document datasets, baselines, and results.
- Caching HF models locally will speed up startup and reduce network dependency.

Makefile Shortcuts
- `make api` – run FastAPI with Uvicorn
- `make cli` – run a demo CLI prediction
- `make test` – run unit tests
- `make freeze` – export exact python deps to `requirements-freeze.txt`

Credits
- HateXplain model: Hate-speech-CNERG/bert-base-uncased-hatexplain
- Libraries: TensorFlow/Keras, PyTorch, Transformers, scikit-learn, pandas, numpy, nltk

