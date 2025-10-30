from typing import Any, Dict
from pydantic import BaseModel
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import pathlib
import sys

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
    outputs = predict_outputs(req.text)
    bert_conf = outputs[1]
    bert_norm = adjust_normal_percentage(bert_conf[1] / 100, bert_conf[0] / 100, bert_conf[2] / 100)

    lstm_conf = outputs[0]
    rf_conf = outputs[2]
    rf_norm = adjust_normal_percentage(rf_conf[2], rf_conf[0], rf_conf[1])

    label, score = dynamic_threshold_prediction(bert_norm, lstm_conf[1], rf_norm)

    return PredictResponse(
        label=label,
        normal_score=score,
        components={
            "lstm": lstm_conf,
            "bert": bert_conf,
            "random_forest": rf_conf,
        },
    )

