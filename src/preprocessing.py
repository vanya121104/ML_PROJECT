import logging
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, List, Dict
from src.config import RANDOM_STATE

logger = logging.getLogger(__name__)

# Constants for column definitions
TARGET_COL = "Attrition"
DROP_COLS = ["EmployeeCount", "Over18", "StandardHours", "EmployeeNumber"]

# Expected schema and details for data dictionary
EXPECTED_CATEGORICAL_VALUES = {
    "BusinessTravel": ["Travel_Rarely", "Travel_Frequently", "Non-Travel"],
    "Department": ["Sales", "Research & Development", "Human Resources"],
    "EducationField": ["Life Sciences", "Other", "Medical", "Marketing", "Technical Degree", "Human Resources"],
    "Gender": ["Female", "Male"],
    "JobRole": [
        "Sales Executive", "Research Scientist", "Laboratory Technician", 
        "Manufacturing Director", "Healthcare Representative", "Manager", 
        "Sales Representative", "Research Director", "Human Resources"
    ],
    "MaritalStatus": ["Single", "Married", "Divorced"],
    "OverTime": ["Yes", "No"]
}

def load_and_validate_data(filepath: str) -> pd.DataFrame:
    """
    Loads raw CSV data and validates it against expected schema.
    """
    logger.info(f"Loading dataset from: {filepath}")
    if not Path(filepath).exists():
        raise FileNotFoundError(f"Dataset not found at path: {filepath}")
        
    df = pd.read_csv(filepath)
    
    # 1. Validate Shape
    if df.empty:
        raise ValueError("Dataset is empty.")
    
    # 2. Check for Target Column
    if TARGET_COL not in df.columns:
        raise ValueError(f"Target column '{TARGET_COL}' is missing.")
        
    # 3. Check for Duplicates (on feature space, ignoring EmployeeNumber if present)
    dup_cols = [c for c in df.columns if c != "EmployeeNumber"]
    num_duplicates = df.duplicated(subset=dup_cols).sum()
    if num_duplicates > 0:
        logger.warning(f"Found {num_duplicates} duplicate rows. Removing duplicates.")
        df = df.drop_duplicates(subset=dup_cols)
        
    return df

def clean_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Cleans the dataframe by dropping redundant features and extracting target.
    """
    logger.info("Cleaning dataset and splitting features from target.")
    df_clean = df.copy()
    
    # Convert target Attrition to binary (1 for Yes, 0 for No)
    if df_clean[TARGET_COL].dtype == object or isinstance(df_clean[TARGET_COL].dtype, pd.StringDtype):
        y = df_clean[TARGET_COL].map({"Yes": 1, "No": 0})
    else:
        y = df_clean[TARGET_COL]
        
    # Drop target and redundant columns
    cols_to_drop = [c for c in DROP_COLS if c in df_clean.columns] + [TARGET_COL]
    X = df_clean.drop(columns=cols_to_drop)
    
    return X, y

def get_feature_types(X: pd.DataFrame) -> Tuple[List[str], List[str]]:
    """
    Identifies numerical and categorical features.
    """
    numerical_features = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = X.select_dtypes(exclude=[np.number]).columns.tolist()
    return numerical_features, categorical_features

def generate_data_dictionary(df: pd.DataFrame, save_path: str = "reports/data_dictionary.md") -> None:
    """
    Generates a Markdown file representing the data dictionary.
    """
    logger.info(f"Generating data dictionary at {save_path}")
    
    markdown_lines = [
        "# HR Attrition Dataset - Data Dictionary",
        "",
        "This document lists all columns in the dataset along with their types, description, and status in the pipeline.",
        "",
        "| Column Name | Data Type | Role | Unique Values / Range | Status |",
        "|-------------|-----------|------|-----------------------|--------|",
    ]
    
    for col in sorted(df.columns):
        dtype_str = str(df[col].dtype)
        
        # Determine Role
        if col == TARGET_COL:
            role = "Target"
            status = "Kept (Encoded as 1/0)"
            unique_info = "Yes (1), No (0)"
        elif col in DROP_COLS:
            role = "Metadata / Redundant"
            status = "Dropped"
            unique_info = "N/A"
        else:
            role = "Feature"
            status = "Kept"
            if col in EXPECTED_CATEGORICAL_VALUES:
                unique_info = ", ".join(EXPECTED_CATEGORICAL_VALUES[col])
            else:
                unique_info = f"Numeric [{df[col].min()} - {df[col].max()}]"
                
        markdown_lines.append(f"| {col} | {dtype_str} | {role} | {unique_info} | {status} |")
        
    # Write to report file
    save_path_p = Path(save_path)
    save_path_p.parent.mkdir(parents=True, exist_ok=True)
    with open(save_path_p, "w", encoding="utf-8") as f:
        f.write("\n".join(markdown_lines))
    
    logger.info("Data dictionary saved successfully.")
