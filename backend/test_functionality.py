#!/usr/bin/env python3
"""
Functionality Test Script for Cyberbullying Detection Project
Tests each component to determine what works and what needs fixing
"""

import sys
import os

def test_imports():
    """Test if all required modules can be imported"""
    print("=" * 50)
    print("TESTING IMPORTS")
    print("=" * 50)
    
    tests = [
        ("pandas", "import pandas as pd"),
        ("numpy", "import numpy as np"),
        ("sklearn", "import sklearn"),
        ("nltk", "import nltk"),
        ("transformers", "import transformers"),
        ("torch", "import torch"),
    ]
    
    working = []
    failed = []
    
    for name, import_cmd in tests:
        try:
            exec(import_cmd)
            print(f"[OK] {name}: OK")
            working.append(name)
        except Exception as e:
            print(f"[FAIL] {name}: FAILED - {str(e)}")
            failed.append(name)
    
    return working, failed

def test_bert():
    """Test BERT model functionality"""
    print("\n" + "=" * 50)
    print("TESTING BERT MODEL")
    print("=" * 50)
    
    try:
        from berttest2 import bert_predict
        test_text = "This is a normal message"
        result = bert_predict(test_text)
        print(f"[OK] BERT Model: OK - Result: {result}")
        return True
    except Exception as e:
        print(f"[FAIL] BERT Model: FAILED - {str(e)}")
        return False

def test_data_loading():
    """Test data loading functionality"""
    print("\n" + "=" * 50)
    print("TESTING DATA LOADING")
    print("=" * 50)
    
    try:
        import pandas as pd
        df = pd.read_csv('train.csv')
        print(f"[OK] Train data: OK - Shape: {df.shape}")
        
        df_test = pd.read_csv('test.csv')
        print(f"[OK] Test data: OK - Shape: {df_test.shape}")
        
        df_twitter = pd.read_csv('twitter_data.csv')
        print(f"[OK] Twitter data: OK - Shape: {df_twitter.shape}")
        
        return True
    except Exception as e:
        print(f"[FAIL] Data loading: FAILED - {str(e)}")
        return False

def test_model_files():
    """Test if model files exist and are accessible"""
    print("\n" + "=" * 50)
    print("TESTING MODEL FILES")
    print("=" * 50)
    
    model_files = [
        'model3.h5',
        'random_forest_model.pkl'
    ]
    
    all_exist = True
    for file in model_files:
        if os.path.exists(file):
            print(f"[OK] {file}: EXISTS")
        else:
            print(f"[FAIL] {file}: MISSING")
            all_exist = False
    
    return all_exist

def test_tensorflow():
    """Test TensorFlow functionality"""
    print("\n" + "=" * 50)
    print("TESTING TENSORFLOW")
    print("=" * 50)
    
    try:
        import tensorflow as tf
        print(f"[OK] TensorFlow: OK - Version: {tf.__version__}")
        
        # Try to load the LSTM model
        model = tf.keras.models.load_model('model3.h5')
        print(f"[OK] LSTM Model: OK - Loaded successfully")
        return True
    except Exception as e:
        print(f"[FAIL] TensorFlow/LSTM: FAILED - {str(e)}")
        return False

def test_random_forest():
    """Test Random Forest functionality"""
    print("\n" + "=" * 50)
    print("TESTING RANDOM FOREST")
    print("=" * 50)
    
    try:
        import joblib
        model = joblib.load('random_forest_model.pkl')
        print(f"[OK] Random Forest Model: OK - Type: {type(model)}")
        return True
    except Exception as e:
        print(f"[FAIL] Random Forest Model: FAILED - {str(e)}")
        return False

def test_ensemble():
    """Test ensemble functionality"""
    print("\n" + "=" * 50)
    print("TESTING ENSEMBLE")
    print("=" * 50)
    
    try:
        # Test individual components that work
        from berttest2 import bert_predict
        
        test_text = "This is a test message"
        bert_result = bert_predict(test_text)
        print(f"[OK] BERT in ensemble: OK - Result: {bert_result}")
        
        # Try to import ensemble functions
        try:
            from ensemble import predict_outputs, dynamic_threshold_prediction
            print("[OK] Ensemble functions: OK - Can be imported")
        except Exception as e:
            print(f"[WARN] Ensemble functions: PARTIAL - {str(e)}")
        
        return True
    except Exception as e:
        print(f"[FAIL] Ensemble: FAILED - {str(e)}")
        return False

def main():
    """Main test function"""
    print("CYBERBULLYING DETECTION PROJECT - FUNCTIONALITY TEST")
    print("=" * 60)
    
    # Run all tests
    working_imports, failed_imports = test_imports()
    bert_works = test_bert()
    data_works = test_data_loading()
    models_exist = test_model_files()
    tf_works = test_tensorflow()
    rf_works = test_random_forest()
    ensemble_works = test_ensemble()
    
    # Summary
    print("\n" + "=" * 60)
    print("FUNCTIONALITY SUMMARY")
    print("=" * 60)
    
    working_components = []
    if bert_works: working_components.append("BERT")
    if data_works: working_components.append("Data Loading")
    if models_exist: working_components.append("Model Files")
    if tf_works: working_components.append("TensorFlow/LSTM")
    if rf_works: working_components.append("Random Forest")
    if ensemble_works: working_components.append("Ensemble (Partial)")
    
    print(f"[OK] WORKING COMPONENTS: {', '.join(working_components)}")
    
    if failed_imports:
        print(f"[FAIL] MISSING DEPENDENCIES: {', '.join(failed_imports)}")
    
    if not tf_works:
        print("[WARN] TensorFlow missing - LSTM model won't work")
    
    if not rf_works:
        print("[WARN] Random Forest model incompatible - needs retraining")
    
    # Overall assessment
    if bert_works and data_works:
        print("\n[SUCCESS] PROJECT STATUS: PARTIALLY FUNCTIONAL")
        print("   - BERT model works perfectly")
        print("   - Data loading works")
        print("   - Some components need fixing")
    else:
        print("\n[ERROR] PROJECT STATUS: NOT FUNCTIONAL")
        print("   - Core components are broken")

if __name__ == "__main__":
    main()
