import logging
import joblib
import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Dict, Any, List
from pathlib import Path
from contextlib import asynccontextmanager

from src.config import MODEL_PATH, setup_logging
from src.explain import get_explainer, get_individual_explanation

setup_logging()
logger = logging.getLogger(__name__)

# Load model package global reference
model_pkg = None
explainer = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global model_pkg, explainer
    logger.info("Initializing API application.")
    if not Path(MODEL_PATH).exists():
        logger.warning(f"Trained model not found at {MODEL_PATH}. Prediction endpoints will fail until trained.")
    else:
        try:
            model_pkg = joblib.load(MODEL_PATH)
            logger.info("Loaded best model package successfully.")
            # Pre-initialize explainer
            pipeline = model_pkg["pipeline"]
            X_train_proc = model_pkg["X_train_proc"]
            explainer, _ = get_explainer(pipeline, X_train_proc)
            logger.info("Initialized SHAP explainer successfully.")
        except Exception as e:
            logger.error(f"Error loading model package or explainer: {e}")
    yield
    logger.info("Shutting down API application.")

app = FastAPI(
    title="Employee Attrition Prediction API",
    description="FastAPI service for predicting employee attrition and explaining predictions using SHAP values.",
    version="1.0.0",
    lifespan=lifespan
)

# Pydantic input schema containing all features expected by the pipeline
class EmployeeInput(BaseModel):
    Age: int = Field(..., example=41)
    BusinessTravel: str = Field(..., example="Travel_Rarely")
    DailyRate: int = Field(..., example=1102)
    Department: str = Field(..., example="Sales")
    DistanceFromHome: int = Field(..., example=1)
    Education: int = Field(..., example=2)
    EducationField: str = Field(..., example="Life Sciences")
    EnvironmentSatisfaction: int = Field(..., example=2)
    Gender: str = Field(..., example="Female")
    HourlyRate: int = Field(..., example=94)
    JobInvolvement: int = Field(..., example=3)
    JobLevel: int = Field(..., example=2)
    JobRole: str = Field(..., example="Sales Executive")
    JobSatisfaction: int = Field(..., example=4)
    MaritalStatus: str = Field(..., example="Single")
    MonthlyIncome: int = Field(..., example=5993)
    MonthlyRate: int = Field(..., example=19479)
    NumCompaniesWorked: int = Field(..., example=8)
    OverTime: str = Field(..., example="Yes")
    PercentSalaryHike: int = Field(..., example=11)
    PerformanceRating: int = Field(..., example=3)
    RelationshipSatisfaction: int = Field(..., example=1)
    StockOptionLevel: int = Field(..., example=0)
    TotalWorkingYears: int = Field(..., example=8)
    TrainingTimesLastYear: int = Field(..., example=0)
    WorkLifeBalance: int = Field(..., example=1)
    YearsAtCompany: int = Field(..., example=6)
    YearsInCurrentRole: int = Field(..., example=4)
    YearsSinceLastPromotion: int = Field(..., example=0)
    YearsWithCurrManager: int = Field(..., example=5)

class PredictionResponse(BaseModel):
    attrition_probability: float
    attrition_prediction: str
    class_label: int

class SHAPFeatureExplanation(BaseModel):
    feature: str
    display_name: str
    shap_value: float
    raw_value: Any

class ExplanationResponse(BaseModel):
    base_value: float
    prediction_probability: float
    feature_explanations: List[SHAPFeatureExplanation]

def check_model_loaded():
    if model_pkg is None:
        raise HTTPException(
            status_code=503, 
            detail="Model is not available. Ensure that the model training process completed."
        )

@app.get("/health")
def health_check() -> Dict[str, str]:
    """
    Checks the status of the service and model loading state.
    """
    status = "healthy"
    model_status = "loaded" if model_pkg is not None else "not_loaded"
    return {"status": status, "model_status": model_status}

@app.post("/predict", response_model=PredictionResponse)
def predict(employee: EmployeeInput):
    """
    Predicts the likelihood of employee attrition.
    """
    check_model_loaded()
    
    # Convert input to DataFrame (1 row)
    input_dict = employee.model_dump()
    df_input = pd.DataFrame([input_dict])
    
    try:
        pipeline = model_pkg["pipeline"]
        prob = float(pipeline.predict_proba(df_input)[0, 1])
        pred_class = int(pipeline.predict(df_input)[0])
        pred_label = "Yes" if pred_class == 1 else "No"
        
        return PredictionResponse(
            attrition_probability=prob,
            attrition_prediction=pred_label,
            class_label=pred_class
        )
    except Exception as e:
        logger.error(f"Error during prediction: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.post("/explain", response_model=ExplanationResponse)
def explain(employee: EmployeeInput):
    """
    Provides local prediction explanations using SHAP values.
    """
    check_model_loaded()
    
    # Convert input to DataFrame (1 row)
    input_dict = employee.model_dump()
    df_input = pd.DataFrame([input_dict])
    
    try:
        pipeline = model_pkg["pipeline"]
        explanation = get_individual_explanation(pipeline, explainer, df_input)
        return ExplanationResponse(**explanation)
    except Exception as e:
        logger.error(f"Error during SHAP explanation: {e}")
        raise HTTPException(status_code=500, detail=f"Explanation error: {str(e)}")
