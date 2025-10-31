from typing import Any, Dict, Optional
from pydantic import BaseModel
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import pathlib
import sys
import traceback
import numpy as np
import io
import os
import tempfile
from PIL import Image

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
    extracted_text: Optional[str] = None
    input_type: str = "text"


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
            input_type="text"
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


# Try to import OCR and Audio libraries (optional dependencies)
try:
    import pytesseract
    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False
    print("Warning: pytesseract not available. Image OCR features disabled.")

try:
    import whisper
    WHISPER_AVAILABLE = True
except ImportError:
    WHISPER_AVAILABLE = False
    print("Warning: whisper not available. Audio transcription features disabled.")


@app.post("/predict/image", response_model=PredictResponse)
async def predict_image(image: UploadFile = File(...)) -> PredictResponse:
    """Analyze image for offensive content by extracting text and analyzing it"""
    if not OCR_AVAILABLE:
        raise HTTPException(
            status_code=503,
            detail="Image OCR not available. Please install pytesseract: pip install pytesseract pillow"
        )
    
    try:
        # Validate file type
        if not image.content_type or not image.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="File must be an image")
        
        # Read image
        contents = await image.read()
        img = Image.open(io.BytesIO(contents))
        
        # Extract text using OCR
        try:
            extracted_text = pytesseract.image_to_string(img)
            extracted_text = extracted_text.strip()
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"OCR failed: {str(e)}. Make sure Tesseract OCR is installed on your system."
            )
        
        if not extracted_text:
            return PredictResponse(
                label="Unknown",
                normal_score=0.5,
                components={},
                extracted_text="No text found in image",
                input_type="image"
            )
        
        # Use existing text prediction
        outputs = predict_outputs(extracted_text)
        bert_conf = outputs[1]
        bert_norm = adjust_normal_percentage(bert_conf[1] / 100, bert_conf[0] / 100, bert_conf[2] / 100)

        lstm_conf = outputs[0]
        rf_conf = outputs[2]
        
        # Convert numpy arrays to lists for JSON serialization
        if isinstance(rf_conf, np.ndarray):
            rf_conf = rf_conf.tolist()
        if isinstance(lstm_conf, np.ndarray):
            lstm_conf = lstm_conf.tolist()
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
            extracted_text=extracted_text,
            input_type="image"
        )
    except HTTPException:
        raise
    except Exception as e:
        error_trace = traceback.format_exc()
        print(f"Error in /predict/image: {error_trace}")
        raise HTTPException(
            status_code=500,
            detail={
                "error": str(e),
                "traceback": error_trace.split('\n')[-5:]
            }
        )


@app.post("/predict/audio", response_model=PredictResponse)
async def predict_audio(audio: UploadFile = File(...)) -> PredictResponse:
    """Analyze audio for offensive content by transcribing and analyzing speech"""
    if not WHISPER_AVAILABLE:
        raise HTTPException(
            status_code=503,
            detail="Audio transcription not available. Please install whisper: pip install openai-whisper"
        )
    
    try:
        # Validate file type
        if not audio.content_type or not audio.content_type.startswith('audio/'):
            raise HTTPException(status_code=400, detail="File must be an audio file")
        
        # Save audio temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as temp_file:
            temp_path = temp_file.name
        
        with open(temp_path, "wb") as f:
            contents = await audio.read()
            f.write(contents)
        
        try:
            # Transcribe audio using Whisper
            model = whisper.load_model("base")
            result = model.transcribe(temp_path)
            extracted_text = result["text"].strip()
        except Exception as e:
            # Clean up temp file
            if os.path.exists(temp_path):
                os.remove(temp_path)
            raise HTTPException(
                status_code=500,
                detail=f"Audio transcription failed: {str(e)}"
            )
        finally:
            # Clean up temp file
            if os.path.exists(temp_path):
                os.remove(temp_path)
        
        if not extracted_text:
            return PredictResponse(
                label="Unknown",
                normal_score=0.5,
                components={},
                extracted_text="No speech detected in audio",
                input_type="audio"
            )
        
        # Use existing text prediction
        outputs = predict_outputs(extracted_text)
        bert_conf = outputs[1]
        bert_norm = adjust_normal_percentage(bert_conf[1] / 100, bert_conf[0] / 100, bert_conf[2] / 100)

        lstm_conf = outputs[0]
        rf_conf = outputs[2]
        
        # Convert numpy arrays to lists for JSON serialization
        if isinstance(rf_conf, np.ndarray):
            rf_conf = rf_conf.tolist()
        if isinstance(lstm_conf, np.ndarray):
            lstm_conf = lstm_conf.tolist()
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
            extracted_text=extracted_text,
            input_type="audio"
        )
    except HTTPException:
        raise
    except Exception as e:
        error_trace = traceback.format_exc()
        print(f"Error in /predict/audio: {error_trace}")
        raise HTTPException(
            status_code=500,
            detail={
                "error": str(e),
                "traceback": error_trace.split('\n')[-5:]
            }
        )

