import os
try:
    from lstmtest3 import lstm_predict
    from tensorflow.keras.models import load_model
    _BASE_DIR = os.path.dirname(__file__)
    _LSTM_PATH = os.path.join(_BASE_DIR, 'model3.h5')
    lstm_model = load_model(_LSTM_PATH)
except Exception as e:
    lstm_predict = None
    lstm_model = None

from berttest2 import bert_predict

##RANDOM FOREST
import joblib
from randomforesttest import randomforestpredict, train_randomforest

model_filename = os.path.join(os.path.dirname(__file__), 'random_forest_model.pkl')
try:
    randomforest_model = joblib.load(model_filename)
except Exception:
    randomforest_model = train_randomforest()

def predict_outputs(text):
    total_output = []
    if lstm_predict is not None and lstm_model is not None:
        total_output.append(lstm_predict(lstm_model, text))
    else:
        # Fallback when LSTM is unavailable (e.g., incompatible TensorFlow/Keras)
        total_output.append([0.5, 0.5])
    total_output.append(bert_predict(text))
    total_output.append(randomforestpredict(randomforest_model , text))
    return total_output


def adjust_normal_percentage(normal_percentage, hate_percentage, offensive_percentage, min_normal_percentage=0.5):
    # Calculate the combined percentage of "Hate" and "Offensive"
    combined_percentage = hate_percentage + offensive_percentage

    # Check if "Normal" has a higher percentage than each of the other classes
    normal_has_priority = normal_percentage > hate_percentage and normal_percentage > offensive_percentage

    # Adjust "Normal" percentage based on priority condition
    if normal_has_priority and normal_percentage >= min_normal_percentage:
        adjusted_normal_percentage = normal_percentage
    elif normal_has_priority and normal_percentage < min_normal_percentage:
        adjusted_normal_percentage = min_normal_percentage
    else:
        adjusted_normal_percentage = combined_percentage / 2

    return adjusted_normal_percentage

def dynamic_threshold_prediction(bert_confidence, lstm_confidence, rf_confidence):
    # Define weights based on priority order
    bert_weight = 0.6  # Increased weight for BERT
    lstm_weight = 0.3  # Decreased weight for LSTM
    rf_weight = 0.1

    total_normal = (bert_weight*bert_confidence) + (lstm_weight*lstm_confidence) + (rf_weight*rf_confidence)
    label = "Normal" if total_normal >= 0.5 else "Cyberbullying"
    print(total_normal)
    print(label)
    return label, float(total_normal)



if __name__ == "__main__":
    input_text = input("Enter text: \n")
    total_output = predict_outputs(input_text)
    print(total_output)
    # Example usage
    bert_confidence = total_output[1]

    bert_confidence = adjust_normal_percentage(bert_confidence[1]/100, bert_confidence[0]/100 , bert_confidence[2]/100)
    lstm_confidence = total_output[0]  # LSTM has Hate and Normal only (or fallback)
    rf_confidence = total_output[2]
    rf_confidence = adjust_normal_percentage(rf_confidence[2],rf_confidence[0] , rf_confidence[1])

    print(bert_confidence//100, lstm_confidence[1], rf_confidence)

    dynamic_threshold_prediction(bert_confidence, lstm_confidence[1], rf_confidence)
