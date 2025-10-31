# 📚 ThreatzShield - Technical Documentation

## Table of Contents
1. [System Architecture](#system-architecture)
2. [Technology Stack](#technology-stack)
3. [Model Details](#model-details)
4. [API Endpoints](#api-endpoints)
5. [Data Flow](#data-flow)
6. [Learning Resources](#learning-resources)
7. [Development Setup](#development-setup)

---

## 🏗️ System Architecture

### High-Level Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    CLIENT LAYER                              │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐     │
│  │  Web Browser  │  │ Mobile App    │  │ CLI Tool     │     │
│  │  (Frontend)   │  │  (Future)     │  │  (cli.py)    │     │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘     │
│         │                  │                  │              │
└─────────┼──────────────────┼──────────────────┼──────────────┘
          │                  │                  │
          │         HTTP/REST API (JSON)        │
          │                  │                  │
┌─────────▼──────────────────▼──────────────────▼──────────────┐
│                    API LAYER (FastAPI)                      │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  Endpoints:                                           │   │
│  │  - POST /predict        (Text Analysis)              │   │
│  │  - POST /predict/image  (Image OCR + Analysis)       │   │
│  │  - POST /predict/audio  (Audio STT + Analysis)       │   │
│  │  - GET  /health         (Health Check)               │   │
│  └──────────────────────────────────────────────────────┘   │
│                          │                                    │
└──────────────────────────┼────────────────────────────────────┘
                           │
┌──────────────────────────▼────────────────────────────────────┐
│              ENSEMBLE ORCHESTRATION LAYER                      │
│  ┌────────────────────────────────────────────────────────┐  │
│  │  ensemble.py - Master Controller                        │  │
│  │  - Coordinates model execution                          │  │
│  │  - Handles fallbacks                                    │  │
│  │  - Aggregates results                                   │  │
│  └────────────────────────────────────────────────────────┘  │
│                          │                                    │
│         ┌────────────────┼────────────────┐                  │
│         │                │                │                  │
└─────────▼────────────────▼────────────────▼──────────────────┘
          │                │                │
┌─────────▼────┐  ┌────────▼────┐  ┌───────▼────────┐
│   BERT       │  │   LSTM       │  │ Random Forest  │
│   Model      │  │   Model      │  │   Model        │
│  (60% weight)│  │ (30% weight) │  │ (10% weight)   │
│              │  │              │  │                │
│  HuggingFace │  │ TensorFlow   │  │ scikit-learn   │
│  Transformers│  │ Keras        │  │                │
│              │  │              │  │                │
│  HateXplain  │  │ Custom LSTM  │  │ TF-IDF + RF    │
│  Pre-trained │  │ Architecture │  │ Classifier     │
└──────────────┘  └───────────────┘  └─────────────────┘
```

---

## 🛠️ Technology Stack

### Backend Technologies

#### **1. FastAPI (Python Web Framework)**
- **Version:** 0.120+
- **Purpose:** REST API framework for handling HTTP requests
- **Key Features:**
  - Automatic API documentation (OpenAPI/Swagger)
  - Async/await support for concurrent requests
  - Type validation with Pydantic
  - Automatic JSON serialization

**Why FastAPI?**
- High performance (comparable to Node.js and Go)
- Built-in data validation
- Automatic interactive API docs
- Easy integration with ML models

**Learn More:**
- 📺 [FastAPI Tutorial - Full Course](https://www.youtube.com/watch?v=tLKKmouUams)
- 📖 [Official FastAPI Documentation](https://fastapi.tiangolo.com/)
- 📺 [Building REST APIs with FastAPI](https://www.youtube.com/watch?v=GN6ICac3OXY)

#### **2. Uvicorn (ASGI Server)**
- **Purpose:** ASGI server to run FastAPI application
- **Features:**
  - Supports HTTP/1.1 and WebSockets
  - Hot reload for development
  - Production-ready with Gunicorn

**Learn More:**
- 📖 [Uvicorn Documentation](https://www.uvicorn.org/)

#### **3. PyTorch & Transformers (BERT Model)**
- **Library:** `transformers` by HuggingFace
- **Model:** `Hate-speech-CNERG/bert-base-uncased-hatexplain`
- **Purpose:** Pre-trained BERT model fine-tuned on hate speech detection
- **Architecture:** 
  - Base: BERT-base-uncased (110M parameters)
  - Fine-tuned on: HateXplain dataset
  - Output: 3-class classification (Hate, Normal, Offensive)

**How BERT Works:**
1. **Tokenization:** Text → Subword tokens (WordPiece)
2. **Embedding:** Tokens → 768-dimensional vectors
3. **Transformer Layers:** 12 layers of self-attention
4. **Classification Head:** Final layer outputs class probabilities

**Learn More:**
- 📺 [BERT Explained - HuggingFace Course](https://www.youtube.com/watch?v=7kLi8u2dJz0)
- 📺 [Transformers Course by HuggingFace](https://www.youtube.com/watch?v=xN6insHnwvM)
- 📖 [HuggingFace Transformers Documentation](https://huggingface.co/docs/transformers)
- 📺 [BERT Model Architecture Explained](https://www.youtube.com/watch?v=xI0HHN5XKDo)

#### **4. TensorFlow/Keras (LSTM Model)**
- **Purpose:** Recurrent neural network for sequence classification
- **Architecture:**
  - **Input:** Tokenized and padded text sequences (maxlen=50)
  - **Embedding Layer:** Converts token IDs to dense vectors
  - **LSTM Layers:** Processes sequential information
  - **Dense Layers:** Final classification output
  - **Output:** Binary classification [Hate probability, Normal probability]

**Preprocessing Pipeline:**
1. Remove emojis using regex
2. Clean text (remove punctuation, numbers)
3. Tokenization (50,000 word vocabulary)
4. Sequence padding (max length: 50 tokens)
5. Convert to NumPy arrays

**Learn More:**
- 📺 [LSTM Networks Explained](https://www.youtube.com/watch?v=YCzL96nL7j0)
- 📺 [TensorFlow and Keras Tutorial](https://www.youtube.com/watch?v=tPYj3fFJGjk)
- 📖 [Keras LSTM Documentation](https://keras.io/api/layers/recurrent_layers/lstm/)
- 📺 [Text Classification with LSTM](https://www.youtube.com/watch?v=wAyznzN7AY0)

#### **5. scikit-learn (Random Forest)**
- **Purpose:** Traditional machine learning classifier
- **Architecture:**
  - **Feature Extraction:** CountVectorizer (TF-IDF alternative)
  - **Model:** Random Forest with 100 estimators
  - **Output:** 3-class probabilities [Hate, Offensive, Normal]
  
**Feature Engineering:**
1. Text cleaning (lowercase, remove URLs, punctuation)
2. Stopword removal (NLTK stopwords)
3. Stemming (Snowball stemmer)
4. CountVectorizer: Text → Bag-of-words vectors

**Learn More:**
- 📺 [Random Forest Algorithm Explained](https://www.youtube.com/watch?v=J4Wdy0Wc_xQ)
- 📺 [scikit-learn Tutorial](https://www.youtube.com/watch?v=pqNCD_5r0IU)
- 📖 [Random Forest Documentation](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html)
- 📺 [Text Classification with scikit-learn](https://www.youtube.com/watch?v=veM2QYIxBgw)

#### **6. Image Processing (OCR)**
- **Library:** `pytesseract` (Python wrapper for Tesseract OCR)
- **Engine:** Tesseract OCR (Google's open-source OCR)
- **Purpose:** Extract text from images for analysis

**How It Works:**
1. Image input (JPG, PNG, GIF, WEBP)
2. Tesseract OCR preprocessing (grayscale, binarization)
3. Text recognition and extraction
4. Extracted text → Sent to ensemble models

**Learn More:**
- 📺 [Tesseract OCR Tutorial](https://www.youtube.com/watch?v=6DjFscX4I_c)
- 📖 [pytesseract Documentation](https://pypi.org/project/pytesseract/)
- 📺 [Image OCR with Python](https://www.youtube.com/watch?v=oXlwWbU8l2o)

#### **7. Audio Processing (Speech-to-Text)**
- **Library:** `openai-whisper`
- **Model:** Whisper Base (OpenAI's speech recognition)
- **Purpose:** Transcribe audio to text for analysis

**How It Works:**
1. Audio input (MP3, WAV, M4A, OGG)
2. Audio preprocessing (resampling, normalization)
3. Whisper model inference
4. Transcription output
5. Transcribed text → Sent to ensemble models

**Learn More:**
- 📺 [OpenAI Whisper Tutorial](https://www.youtube.com/watch?v=Y97gCc1Z2ag)
- 📖 [Whisper Documentation](https://github.com/openai/whisper)
- 📺 [Speech-to-Text with Whisper](https://www.youtube.com/watch?v=0G3-FNaG5HQ)

### Frontend Technologies

#### **1. HTML5, CSS3, JavaScript**
- **No Frameworks:** Pure vanilla JavaScript for simplicity
- **CSS Features:**
  - Flexbox for layout
  - CSS animations (glow effects, slide-in)
  - Responsive design (media queries)
  - WhatsApp-style chat interface

**Learn More:**
- 📺 [Modern JavaScript Tutorial](https://www.youtube.com/watch?v=hdI2bqOjy3c)
- 📺 [CSS Animations Tutorial](https://www.youtube.com/watch?v=jgw82b5Y2SY)
- 📺 [Build WhatsApp Clone](https://www.youtube.com/watch?v=9J0h-0bQ3rM)

---

## 🧠 Model Details

### 1. BERT Model (60% Weight)

**Architecture:**
- **Base Model:** BERT-base-uncased
- **Fine-tuning:** HateXplain dataset
- **Input:** Raw text string
- **Processing:**
  1. Tokenization with WordPiece
  2. Add [CLS] and [SEP] tokens
  3. Convert to token IDs
  4. Pass through 12-layer transformer
  5. [CLS] token embedding → Classification head
- **Output:** [Hate%, Normal%, Offensive%] (percentages sum to 100)

**Code Location:** `cyber_detect_backend-master/berttest2.py`

**Key Function:**
```python
def bert_predict(text):
    # Load tokenizer and model
    tokenizer, model = _load_hf_model()
    
    # Tokenize input
    inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True)
    
    # Forward pass
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Get probabilities
    probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
    
    return [hate%, normal%, offensive%]
```

**Why 60% Weight?**
- Most accurate on benchmark datasets
- Pre-trained on large corpus
- Best context understanding

### 2. LSTM Model (30% Weight)

**Architecture:**
- **Type:** Bidirectional LSTM
- **Input:** Padded sequences (maxlen=50)
- **Vocabulary:** 50,000 words
- **Embedding:** Learned embeddings
- **LSTM Layers:** Multiple LSTM layers for sequence processing
- **Output:** [Hate probability, Normal probability] (binary, sum to 1)

**Preprocessing Steps:**
1. Remove emojis
2. Clean text (lowercase, remove punctuation, numbers)
3. Filter words (length > 3)
4. Tokenization
5. Sequence padding/truncation

**Code Location:** `cyber_detect_backend-master/lstmtest3.py`

**Key Function:**
```python
def lstm_predict(model, text):
    # Clean text
    cleaned = clean_text(remove_emoji(text))
    
    # Tokenize
    sequences = tokenizer.texts_to_sequences([cleaned])
    
    # Pad sequences
    padded = pad_sequences(sequences, maxlen=50)
    
    # Predict
    predictions = model.predict(padded)
    
    return [hate_prob, normal_prob]
```

**Why 30% Weight?**
- Good at sequential patterns
- Captures word order dependencies
- Less accurate than BERT but faster

### 3. Random Forest (10% Weight)

**Architecture:**
- **Features:** Bag-of-words (CountVectorizer)
- **Estimators:** 100 decision trees
- **Text Preprocessing:**
  1. Lowercase
  2. Remove URLs, HTML tags
  3. Remove punctuation
  4. Remove stopwords
  5. Stemming (Snowball stemmer)
- **Output:** [Hate prob, Offensive prob, Normal prob]

**Code Location:** `cyber_detect_backend-master/randomforesttest.py`

**Key Function:**
```python
def randomforestpredict(model, text):
    # Clean text
    cleaned = clean(text)
    
    # Vectorize
    vectorized = cv.transform([cleaned])
    
    # Predict probabilities
    probabilities = model.predict_proba(vectorized)
    
    return [hate, offensive, normal]
```

**Why 10% Weight?**
- Fast inference
- Good baseline
- Less sophisticated than neural networks

### 4. Ensemble Aggregation

**Location:** `cyber_detect_backend-master/ensemble.py`

**Process:**
1. Get predictions from all three models
2. Normalize BERT output (percentages to 0-1 range)
3. Adjust normal percentage for BERT and RF
4. Weighted combination:
   ```python
   total_normal = (0.6 × bert_normal) + (0.3 × lstm_normal) + (0.1 × rf_normal)
   ```
5. Threshold at 0.5:
   - `total_normal >= 0.5` → **Normal**
   - `total_normal < 0.5` → **Cyberbullying**

**Key Functions:**
- `predict_outputs(text)`: Get predictions from all models
- `adjust_normal_percentage()`: Normalize probabilities
- `dynamic_threshold_prediction()`: Final classification

---

## 🔌 API Endpoints

### 1. Health Check
```http
GET /health
```
**Response:**
```json
{
  "status": "ok"
}
```

### 2. Text Prediction
```http
POST /predict
Content-Type: application/json

{
  "text": "Your text to analyze"
}
```

**Response:**
```json
{
  "label": "Normal",
  "normal_score": 0.85,
  "components": {
    "bert": [5.2, 75.3, 19.5],
    "lstm": [0.15, 0.85],
    "random_forest": [0.10, 0.20, 0.70]
  },
  "input_type": "text"
}
```

### 3. Image Prediction
```http
POST /predict/image
Content-Type: multipart/form-data

Form Data:
  image: [binary file]
```

**Response:**
```json
{
  "label": "Cyberbullying",
  "normal_score": 0.35,
  "components": { ... },
  "extracted_text": "Text extracted from image",
  "input_type": "image"
}
```

### 4. Audio Prediction
```http
POST /predict/audio
Content-Type: multipart/form-data

Form Data:
  audio: [binary file]
```

**Response:**
```json
{
  "label": "Normal",
  "normal_score": 0.72,
  "components": { ... },
  "extracted_text": "Transcribed audio text",
  "input_type": "audio"
}
```

---

## 📊 Data Flow

### Text Analysis Flow

```
User Input (Text)
    │
    ▼
FastAPI Endpoint (/predict)
    │
    ▼
ensemble.py → predict_outputs(text)
    │
    ├──► berttest2.py → bert_predict(text)
    │    └──► HuggingFace Transformers
    │         └──► BERT Model Inference
    │              └──► [Hate%, Normal%, Offensive%]
    │
    ├──► lstmtest3.py → lstm_predict(model, text)
    │    └──► Text Preprocessing
    │         └──► Tokenization
    │              └──► LSTM Model Inference
    │                   └──► [Hate prob, Normal prob]
    │
    └──► randomforesttest.py → randomforestpredict(model, text)
         └──► Text Cleaning & Stemming
              └──► CountVectorizer
                   └──► Random Forest Inference
                        └──► [Hate, Offensive, Normal probs]
    │
    ▼
ensemble.py → adjust_normal_percentage() (for BERT & RF)
    │
    ▼
ensemble.py → dynamic_threshold_prediction()
    │
    ▼
Weighted Aggregation: 0.6×BERT + 0.3×LSTM + 0.1×RF
    │
    ▼
Final Label & Confidence Score
    │
    ▼
JSON Response → Frontend
```

### Image Analysis Flow

```
User Upload (Image)
    │
    ▼
FastAPI Endpoint (/predict/image)
    │
    ▼
PIL Image Processing
    │
    ▼
pytesseract → OCR
    │
    ▼
Extracted Text
    │
    ▼
[Same as Text Analysis Flow]
```

### Audio Analysis Flow

```
User Upload (Audio)
    │
    ▼
FastAPI Endpoint (/predict/audio)
    │
    ▼
Save to Temporary File
    │
    ▼
Whisper Model → Speech-to-Text
    │
    ▼
Transcribed Text
    │
    ▼
[Same as Text Analysis Flow]
```

---

## 🎓 Learning Resources

### Machine Learning Fundamentals

#### **BERT & Transformers**
- 📺 [BERT: Pre-training of Deep Bidirectional Transformers](https://www.youtube.com/watch?v=Ux4cg3MsNaE)
- 📺 [The Illustrated Transformer](https://www.youtube.com/watch?v=4Bdc55j80l8)
- 📺 [HuggingFace NLP Course](https://huggingface.co/course)
- 📺 [Fine-tuning BERT for Text Classification](https://www.youtube.com/watch?v=Kb2XNcC07qo)

#### **LSTM & Recurrent Neural Networks**
- 📺 [Understanding LSTM Networks](https://www.youtube.com/watch?v=YCzL96nL7j0)
- 📺 [RNN, LSTM, GRU Explained](https://www.youtube.com/watch?v=9zhrxE5PQgY)
- 📺 [Sequence Models by Andrew Ng](https://www.youtube.com/watch?v=ySEx_Bqxvvo)
- 📺 [Text Classification with LSTM](https://www.youtube.com/watch?v=wAyznzN7AY0)

#### **Random Forest & Traditional ML**
- 📺 [Random Forest Algorithm - StatQuest](https://www.youtube.com/watch?v=J4Wdy0Wc_xQ)
- 📺 [Decision Trees Explained](https://www.youtube.com/watch?v=_L39rN6gz7Y)
- 📺 [Ensemble Methods in Machine Learning](https://www.youtube.com/watch?v=Un9zObFjBH0)
- 📺 [scikit-learn Machine Learning Course](https://www.youtube.com/watch?v=pqNCD_5r0IU)

### NLP & Text Processing

- 📺 [Natural Language Processing Course](https://www.youtube.com/watch?v=fM4qRMfF3nw)
- 📺 [Text Preprocessing Techniques](https://www.youtube.com/watch?v=VeXqKX7q1fY)
- 📺 [Tokenization Explained](https://www.youtube.com/watch?v=MRc5LEvqkPs)
- 📺 [Word Embeddings Tutorial](https://www.youtube.com/watch?v=5MaWmXwxFNQ)

### Deep Learning

- 📺 [Deep Learning Specialization (Coursera)](https://www.youtube.com/watch?v=CS4cs9xVecg)
- 📺 [Neural Networks from Scratch](https://www.youtube.com/watch?v=aircAruvnKk)
- 📺 [PyTorch for Deep Learning](https://www.youtube.com/watch?v=V_xro1bcAuA)
- 📺 [TensorFlow 2.0 Complete Course](https://www.youtube.com/watch?v=tPYj3fFJGjk)

### FastAPI & Web Development

- 📺 [FastAPI Tutorial for Beginners](https://www.youtube.com/watch?v=tLKKmouUams)
- 📺 [Building REST APIs with FastAPI](https://www.youtube.com/watch?v=GN6ICac3OXY)
- 📺 [FastAPI + Machine Learning Deployment](https://www.youtube.com/watch?v=3ecNu9NCW6w)
- 📖 [FastAPI Best Practices](https://fastapi.tiangolo.com/tutorial/)

### Computer Vision (OCR)

- 📺 [OCR with Python and Tesseract](https://www.youtube.com/watch?v=6DjFscX4I_c)
- 📺 [Image Processing with Python](https://www.youtube.com/watch?v=oXlwWbU8l2o)
- 📖 [Tesseract OCR Documentation](https://tesseract-ocr.github.io/)

### Speech Recognition

- 📺 [OpenAI Whisper Tutorial](https://www.youtube.com/watch?v=Y97gCc1Z2ag)
- 📺 [Speech-to-Text Applications](https://www.youtube.com/watch?v=0G3-FNaG5HQ)
- 📺 [Audio Processing with Python](https://www.youtube.com/watch?v=W4TSN7qJXEQ)

### System Design & Architecture

- 📺 [System Design Interview Prep](https://www.youtube.com/watch?v=Un16BHF2eCE)
- 📺 [ML System Design](https://www.youtube.com/watch?v=YiaYZxhr2zI)
- 📺 [Microservices Architecture](https://www.youtube.com/watch?v=j6ow-UemzBc)

### Deployment & DevOps

- 📺 [Docker Tutorial for Beginners](https://www.youtube.com/watch?v=fqMOX6JJhGo)
- 📺 [FastAPI Production Deployment](https://www.youtube.com/watch?v=3ecNu9NCW6w)
- 📺 [CI/CD with GitHub Actions](https://www.youtube.com/watch?v=mFFeXtmrAeQ)

---

## 💻 Development Setup

### Prerequisites

```bash
# Python 3.11+
python --version

# Install system dependencies
# Windows: Download Tesseract OCR installer
# Linux: sudo apt-get install tesseract-ocr
# Mac: brew install tesseract
```

### Installation Steps

```bash
# 1. Clone repository
git clone <repository-url>
cd Cyberdetect-master

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 3. Install Python dependencies
pip install -r cyber_detect_backend-master/requirements.txt
pip install fastapi uvicorn python-multipart
pip install pytesseract openai-whisper

# 4. Download NLTK data (if needed)
python -c "import nltk; nltk.download('stopwords')"

# 5. Start backend server
uvicorn api:app --reload --host 0.0.0.0 --port 8000

# 6. Open frontend
# Open frontend/index.html in your browser
```

### Model Files

**Note:** Large model files are gitignored. Download from:
- BERT: Automatically downloaded from HuggingFace on first use
- LSTM: `model3.h5` (train using `scripts/train_lstm.py`)
- Random Forest: `random_forest_model.pkl` (auto-generated if missing)

---

## 🔍 Code Walkthrough

### Key Files Explained

#### `api.py` - Main API Server
- FastAPI application setup
- CORS middleware configuration
- Three prediction endpoints
- Error handling and logging

#### `cyber_detect_backend-master/ensemble.py` - Model Orchestrator
- Coordinates all three models
- Handles model fallbacks
- Aggregates results with weights
- Dynamic threshold prediction

#### `cyber_detect_backend-master/berttest2.py` - BERT Integration
- Loads HuggingFace model
- Tokenization and inference
- Fallback mechanism

#### `cyber_detect_backend-master/lstmtest3.py` - LSTM Integration
- Text preprocessing pipeline
- Tokenization and sequence padding
- LSTM model inference

#### `cyber_detect_backend-master/randomforesttest.py` - RF Integration
- Text cleaning and stemming
- CountVectorizer feature extraction
- Random Forest prediction

---

## 📈 Performance Optimization

### Current Optimizations

1. **Model Caching:** Models loaded once at startup
2. **Async Endpoints:** FastAPI async support for concurrent requests
3. **Fallback Mechanisms:** System remains operational if one model fails
4. **JSON Serialization:** NumPy types converted to native Python

### Future Optimizations

- Batch prediction endpoint
- Model quantization
- GPU acceleration
- Caching frequently analyzed text
- Rate limiting

---

## 🧪 Testing

### Run Tests

```bash
# Unit tests
python -m unittest discover -s tests

# API integration tests
python tests/test_api.py

# Model evaluation
python tests/evaluate_models.py
```

---

## 📝 Additional Notes

### Model Weights Rationale

- **BERT (60%):** Most accurate, best context understanding
- **LSTM (30%):** Good sequential patterns, complementary to BERT
- **Random Forest (10%):** Baseline, fast, adds diversity

### Threshold Logic

- Dynamic threshold at 0.5 for binary classification
- Confidence scores range from 0.0 to 1.0
- Higher scores = More likely to be safe content

### Error Handling

- Graceful fallbacks if models fail to load
- Returns neutral predictions (50/50) if model unavailable
- Comprehensive error logging

---

**Last Updated:** 2025-01-30  
**Version:** 1.0

