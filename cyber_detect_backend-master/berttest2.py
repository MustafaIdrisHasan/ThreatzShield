import os
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification


def _load_hf_model():
    """Try local model dir first, then fall back to HF hub.
    If both fail (e.g., offline), return None, None.
    """
    local_dir = os.getenv("HATE_MODEL_DIR", os.path.join(os.path.dirname(__file__), "models", "hatexplain"))
    try:
        if os.path.isdir(local_dir):
            tokenizer = AutoTokenizer.from_pretrained(local_dir)
            model = AutoModelForSequenceClassification.from_pretrained(local_dir)
            return tokenizer, model
    except Exception:
        pass

    try:
        tokenizer = AutoTokenizer.from_pretrained("Hate-speech-CNERG/bert-base-uncased-hatexplain")
        model = AutoModelForSequenceClassification.from_pretrained("Hate-speech-CNERG/bert-base-uncased-hatexplain")
        return tokenizer, model
    except Exception:
        return None, None


def bert_predict(text):

    classes = {0: "hate speech", 1: "normal", 2: "offensive language"}

    tokenizer, model = _load_hf_model()
    if tokenizer is None or model is None:
        # Offline fallback: uniform distribution
        probs_list = [33.33, 33.33, 33.33]
        print("BERT model unavailable, using fallback distribution")
    else:
        texts = [text]
        inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True)
        with torch.no_grad():
            outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
        probs_list = probs.squeeze().tolist()
        probs_list = [p * 100 for p in probs_list]

    class_probabilities = {classes[i]: probs_list[i] for i in range(len(probs_list))}

    print("Text:", text)
    print("Class Probabilities:", class_probabilities)

    return list(class_probabilities.values())
