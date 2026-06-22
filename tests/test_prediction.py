import pytest
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from src.preprocessing import clean_data, get_feature_types
from src.pipeline import build_preprocessor, create_full_pipeline

@pytest.fixture
def dummy_trained_pipeline():
    """
    Fits a simple pipeline on mock training data for prediction testing.
    """
    raw_df = pd.DataFrame({
        "Age": [41, 49, 37, 33, 27, 32, 59, 30, 38, 36] * 3,
        "Attrition": ["Yes", "No", "Yes", "No", "No", "No", "No", "No", "No", "No"] * 3,
        "BusinessTravel": ["Travel_Rarely", "Travel_Frequently", "Travel_Rarely", "Travel_Frequently", "Travel_Rarely", 
                           "Travel_Frequently", "Travel_Rarely", "Travel_Rarely", "Travel_Frequently", "Travel_Rarely"] * 3,
        "EmployeeCount": [1] * 30,
        "EmployeeNumber": list(range(1, 31)),
        "Over18": ["Y"] * 30,
        "StandardHours": [80] * 30,
        "MonthlyIncome": [5993, 5130, 2090, 2909, 3468, 3068, 2670, 2693, 9526, 5237] * 3,
        "Gender": ["Female", "Male", "Male", "Female", "Male", "Male", "Female", "Male", "Male", "Male"] * 3,
        "OverTime": ["Yes", "No", "Yes", "Yes", "No", "No", "Yes", "No", "No", "No"] * 3
    })
    
    X, y = clean_data(raw_df)
    numeric, categorical = get_feature_types(X)
    
    preprocessor = build_preprocessor(numeric, categorical)
    model = DecisionTreeClassifier(random_state=42)
    
    # Create pipeline (disable SMOTE to avoid requiring library if not installed yet during dummy test)
    pipeline = create_full_pipeline(preprocessor, model, use_smote=False)
    pipeline.fit(X, y)
    
    return pipeline

def test_pipeline_prediction(dummy_trained_pipeline):
    # Test sample
    single_input = pd.DataFrame([{
        "Age": 35,
        "BusinessTravel": "Travel_Rarely",
        "MonthlyIncome": 4000,
        "Gender": "Male",
        "OverTime": "Yes"
    }])
    
    # Predict probabilities
    prob = dummy_trained_pipeline.predict_proba(single_input)
    assert prob.shape == (1, 2)
    assert 0.0 <= prob[0, 1] <= 1.0
    
    # Predict classes
    pred = dummy_trained_pipeline.predict(single_input)
    assert pred.shape == (1,)
    assert pred[0] in [0, 1]
