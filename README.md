Cyberbullying Detection – Quick Start

Overview
- Ensemble classifier combining an LSTM (Keras), a BERT model, and a RandomForest to detect hate/offensive language vs. normal text.
- Robust fallbacks allow running offline: LSTM/RandomForest models are loaded from local files; BERT falls back to a uniform prior when the HuggingFace model is unavailable.

Project Layout
- Backend code: `cyber_detect_backend-master/`
- CLI: `cli.py`
- API: `api.py` (FastAPI)
- Models and data (present in repo): `cyber_detect_backend-master/model3.h5`, `cyber_detect_backend-master/random_forest_model.pkl`, `cyber_detect_backend-master/twitter_data.csv`

Prerequisites
- Windows: Use the bundled virtual environment for a zero-install run
  - Python path: `cyber_detect_backend-master/.venv/Scripts/python.exe`
  - Uvicorn path: `cyber_detect_backend-master/.venv/Scripts/uvicorn.exe`
- Or install dependencies (Linux/Mac/WSL):
  - Minimal: `pip install -r cyber_detect_backend-master/requirements.txt`
  - Full (exact versions from the bundled venv): see `requirements-freeze.txt`

Run (CLI)
- Example: `cyber_detect_backend-master/.venv/Scripts/python.exe cli.py "Your text here"`

Run (API)
- Start server: `cyber_detect_backend-master/.venv/Scripts/uvicorn.exe api:app --reload`
- Predict: `curl -X POST http://127.0.0.1:8000/predict -H "Content-Type: application/json" -d "{\"text\": \"Hello world\"}"`

Notes on Fallbacks
- BERT (`berttest2.py`) tries a local model directory first (`cyber_detect_backend-master/models/hatexplain` or `HATE_MODEL_DIR`), then the HuggingFace hub. If neither is available (offline), it returns a uniform distribution across classes.
- NLTK stopwords are optional; if unavailable, a small built-in list is used to keep preprocessing functional.

Reproducibility
- The repository includes `cyber_detect_backend-master/requirements.txt` (curated) and `requirements-freeze.txt` (full lock from the bundled venv). Prefer the curated file on fresh setups; use the freeze when you need to replicate the exact environment.

Testing
- Quick functionality test: `cyber_detect_backend-master/.venv/Scripts/python.exe cyber_detect_backend-master/test_functionality.py`
- Unit tests: `cyber_detect_backend-master/.venv/Scripts/python.exe -m unittest discover -s tests -p "*_test.py"`

