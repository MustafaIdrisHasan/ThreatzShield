"""
Evaluate model performance on test dataset
"""
import sys
import pathlib
import time

BASE_DIR = pathlib.Path(__file__).resolve().parents[1]
BACKEND_DIR = BASE_DIR / "cyber_detect_backend-master"
sys.path.append(str(BACKEND_DIR))

from ensemble import predict_outputs, adjust_normal_percentage, dynamic_threshold_prediction

# Test dataset - mix of safe and harmful examples
TEST_DATASET = [
    # Safe examples
    ("Hello, how are you today?", "normal"),
    ("I love programming and learning new things!", "normal"),
    ("The weather is nice today.", "normal"),
    ("Can we schedule a meeting for tomorrow?", "normal"),
    ("Thank you for your help!", "normal"),
    ("Great job on the project!", "normal"),
    ("Looking forward to our collaboration.", "normal"),
    ("This is a very informative article.", "normal"),
    
    # Potentially harmful examples
    ("You are stupid and should go away", "harmful"),
    ("I hate you, you worthless piece of garbage", "harmful"),
    ("You are a terrible person and nobody likes you", "harmful"),
    ("Get lost, you idiot", "harmful"),
    ("You are so dumb, it's unbelievable", "harmful"),
    ("Nobody cares about your opinion, shut up", "harmful"),
]


def evaluate():
    """Evaluate model performance"""
    print("=" * 60)
    print("MODEL EVALUATION")
    print("=" * 60)
    print()
    
    correct = 0
    total = len(TEST_DATASET)
    predictions = []
    response_times = []
    
    print(f"Testing on {total} examples...")
    print()
    
    for text, expected_label in TEST_DATASET:
        start_time = time.time()
        
        try:
            outputs = predict_outputs(text)
            bert_conf = outputs[1]
            bert_norm = adjust_normal_percentage(
                bert_conf[1] / 100, bert_conf[0] / 100, bert_conf[2] / 100
            )
            
            lstm_conf = outputs[0]
            rf_conf = outputs[2]
            
            # Convert numpy arrays
            import numpy as np
            if isinstance(rf_conf, np.ndarray):
                rf_conf = rf_conf.tolist()
            if isinstance(lstm_conf, np.ndarray):
                lstm_conf = lstm_conf.tolist()
            
            rf_norm = adjust_normal_percentage(rf_conf[2], rf_conf[0], rf_conf[1])
            label, score = dynamic_threshold_prediction(bert_norm, lstm_conf[1], rf_norm)
            
            elapsed = time.time() - start_time
            response_times.append(elapsed)
            
            # Convert to binary classification
            predicted = "normal" if label.lower() == "normal" else "harmful"
            expected_binary = expected_label  # Already binary
            
            is_correct = predicted == expected_binary
            if is_correct:
                correct += 1
            
            predictions.append({
                "text": text[:50] + "..." if len(text) > 50 else text,
                "expected": expected_binary,
                "predicted": predicted,
                "score": score,
                "correct": is_correct,
                "time": elapsed
            })
            
            status = "✓" if is_correct else "✗"
            print(f"{status} [{predicted:8s}] (expected: {expected_binary:8s}) | Score: {score:.3f} | {elapsed:.2f}s")
            
        except Exception as e:
            print(f"✗ ERROR: {str(e)}")
            predictions.append({
                "text": text[:50],
                "expected": expected_binary,
                "predicted": "error",
                "correct": False
            })
    
    print()
    print("=" * 60)
    print("RESULTS")
    print("=" * 60)
    
    accuracy = (correct / total) * 100
    print(f"Total Examples: {total}")
    print(f"Correct: {correct}")
    print(f"Accuracy: {accuracy:.2f}%")
    print()
    
    # Calculate precision, recall, F1
    true_positives = sum(1 for p in predictions if p["predicted"] == "harmful" and p["expected"] == "harmful")
    false_positives = sum(1 for p in predictions if p["predicted"] == "harmful" and p["expected"] == "normal")
    false_negatives = sum(1 for p in predictions if p["predicted"] == "normal" and p["expected"] == "harmful")
    
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    print(f"Precision (Harmful): {precision:.3f}")
    print(f"Recall (Harmful): {recall:.3f}")
    print(f"F1-Score: {f1:.3f}")
    print()
    
    if response_times:
        avg_time = sum(response_times) / len(response_times)
        print(f"Average Response Time: {avg_time:.2f}s")
        print(f"Min Response Time: {min(response_times):.2f}s")
        print(f"Max Response Time: {max(response_times):.2f}s")
    
    print()
    print("=" * 60)
    
    return accuracy, precision, recall, f1


if __name__ == "__main__":
    evaluate()


