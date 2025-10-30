from typing import Any, Dict
from pydantic import BaseModel
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import pathlib
import sys
import traceback
import numpy as np

BASE_DIR = pathlib.Path(__file__).parent
BACKEND_DIR = BASE_DIR / "cyber_detect_backend-master"
sys.path.append(str(BACKEND_DIR))

from ensemble import (
    predict_outputs,
    adjust_normal_percentage,
    dynamic_threshold_prediction,
)


class PredictRequest(BaseModel):
    text: str


class PredictResponse(BaseModel):
    label: str
    normal_score: float
    components: Dict[str, Any]


app = FastAPI(title="Cyberbullying Detection API")

# Allow local dev pages to call the API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest) -> PredictResponse:
    try:
        outputs = predict_outputs(req.text)
        bert_conf = outputs[1]
        bert_norm = adjust_normal_percentage(bert_conf[1] / 100, bert_conf[0] / 100, bert_conf[2] / 100)

        lstm_conf = outputs[0]
        rf_conf = outputs[2]
        
        # Convert numpy arrays to lists for JSON serialization
        if isinstance(rf_conf, np.ndarray):
            rf_conf = rf_conf.tolist()
        if isinstance(lstm_conf, np.ndarray):
            lstm_conf = lstm_conf.tolist()
        # Convert numpy scalars to native Python types
        rf_conf = [float(x) if isinstance(x, (np.generic, np.number)) else x for x in rf_conf]
        lstm_conf = [float(x) if isinstance(x, (np.generic, np.number)) else x for x in lstm_conf]
        bert_conf = [float(x) if isinstance(x, (np.generic, np.number)) else x for x in bert_conf]
        
        rf_norm = adjust_normal_percentage(rf_conf[2], rf_conf[0], rf_conf[1])

        label, score = dynamic_threshold_prediction(bert_norm, lstm_conf[1], rf_norm)

        return PredictResponse(
            label=str(label),
            normal_score=float(score),
            components={
                "lstm": lstm_conf,
                "bert": bert_conf,
                "random_forest": rf_conf,
            },
        )
    except Exception as e:
        error_trace = traceback.format_exc()
        print(f"Error in /predict: {error_trace}")  # Print to server logs
        raise HTTPException(
            status_code=500,
            detail={
                "error": str(e),
                "traceback": error_trace.split('\n')[-5:]  # Last 5 lines
            }
        )

