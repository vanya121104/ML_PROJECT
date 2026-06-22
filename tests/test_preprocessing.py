import pytest
import pandas as pd
import numpy as np
from src.preprocessing import clean_data, get_feature_types, load_and_validate_data

@pytest.fixture
def sample_raw_data():
    """
    Creates a sample dataframe containing all structure and edge cases.
    """
    return pd.DataFrame({
        "Age": [41, 49, 37],
        "Attrition": ["Yes", "No", "Yes"],
        "BusinessTravel": ["Travel_Rarely", "Travel_Frequently", "Travel_Rarely"],
        "EmployeeCount": [1, 1, 1],
        "EmployeeNumber": [1, 2, 4],
        "Over18": ["Y", "Y", "Y"],
        "StandardHours": [80, 80, 80],
        "MonthlyIncome": [5993, 5130, 2090]
    })

def test_clean_data(sample_raw_data):
    X, y = clean_data(sample_raw_data)
    
    # Check shape
    assert X.shape == (3, 3)
    assert len(y) == 3
    
    # Check dropped columns
    dropped_cols = ["EmployeeCount", "Over18", "StandardHours", "EmployeeNumber", "Attrition"]
    for col in dropped_cols:
        assert col not in X.columns
        
    # Check target encoding
    assert list(y) == [1, 0, 1]

def test_get_feature_types(sample_raw_data):
    X, _ = clean_data(sample_raw_data)
    numeric, categorical = get_feature_types(X)
    
    assert "Age" in numeric
    assert "MonthlyIncome" in numeric
    assert "BusinessTravel" in categorical
    assert len(numeric) == 2
    assert len(categorical) == 1
