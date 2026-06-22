import logging
import json
import joblib
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC

# Dynamic imports for gradient boosted libraries
try:
    from xgboost import XGBClassifier
    xgb_available = True
except ImportError:
    xgb_available = False

try:
    from lightgbm import LGBMClassifier
    lgbm_available = True
except ImportError:
    lgbm_available = False

try:
    from catboost import CatBoostClassifier
    catboost_available = True
except ImportError:
    catboost_available = False

from src.config import (
    RAW_DATA_PATH, MODEL_PATH, FIGURES_DIR, RANDOM_STATE, 
    TEST_SIZE, METRICS_PATH, setup_logging
)
from src.preprocessing import (
    load_and_validate_data, clean_data, get_feature_types, 
    generate_data_dictionary
)
from src.pipeline import build_preprocessor, create_full_pipeline
from src.evaluate import evaluate_model
from src.explain import get_explainer, save_global_shap_summary

# Setup logging
setup_logging()
logger = logging.getLogger(__name__)

def train_and_select_best_model():
    # 1. Ingest and Validate Data
    df = load_and_validate_data(RAW_DATA_PATH)
    
    # Generate Data Dictionary
    generate_data_dictionary(df)
    
    # Clean data & separate features/target
    X, y = clean_data(df)
    
    # 2. Train-Test Split ( stratified to maintain target ratio )
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    
    logger.info(f"Train size: {X_train.shape}, Test size: {X_test.shape}")
    
    # Get feature lists
    numerical_features, categorical_features = get_feature_types(X_train)
    logger.info(f"Numerical features: {len(numerical_features)}, Categorical features: {len(categorical_features)}")
    
    # Build common preprocessor
    preprocessor = build_preprocessor(numerical_features, categorical_features)
    
    # 3. Model definition and tuning parameters
    models_to_tune = {}
    
    # Logistic Regression
    models_to_tune["Logistic Regression"] = {
        "model": LogisticRegression(random_state=RANDOM_STATE, max_iter=1000),
        "params": {
            "classifier__C": np.logspace(-3, 2, 20),
            "classifier__solver": ["liblinear", "lbfgs"]
        }
    }
    
    # Random Forest
    models_to_tune["Random Forest"] = {
        "model": RandomForestClassifier(random_state=RANDOM_STATE),
        "params": {
            "classifier__n_estimators": [100, 200, 300],
            "classifier__max_depth": [None, 5, 10, 15],
            "classifier__min_samples_split": [2, 5, 10],
            "classifier__class_weight": [None, "balanced"]
        }
    }
    
    # Gradient Boosting
    models_to_tune["Gradient Boosting"] = {
        "model": GradientBoostingClassifier(random_state=RANDOM_STATE),
        "params": {
            "classifier__n_estimators": [100, 200],
            "classifier__learning_rate": [0.01, 0.05, 0.1, 0.2],
            "classifier__max_depth": [3, 4, 5]
        }
    }
    
    # Support Vector Machine (SVC)
    models_to_tune["SVM"] = {
        "model": SVC(random_state=RANDOM_STATE, probability=True),
        "params": {
            "classifier__C": [0.1, 1, 10],
            "classifier__kernel": ["rbf", "linear"]
        }
    }
    
    # Conditionally add gradient boosted models
    if xgb_available:
        models_to_tune["XGBoost"] = {
            "model": XGBClassifier(random_state=RANDOM_STATE, eval_metric="logloss"),
            "params": {
                "classifier__n_estimators": [100, 200],
                "classifier__max_depth": [3, 5, 7],
                "classifier__learning_rate": [0.01, 0.05, 0.1, 0.2]
            }
        }
        
    if lgbm_available:
        models_to_tune["LightGBM"] = {
            "model": LGBMClassifier(random_state=RANDOM_STATE, verbose=-1),
            "params": {
                "classifier__n_estimators": [100, 200],
                "classifier__max_depth": [3, 5, 7],
                "classifier__learning_rate": [0.01, 0.05, 0.1],
                "classifier__class_weight": [None, "balanced"]
            }
        }
        
    if catboost_available:
        models_to_tune["CatBoost"] = {
            "model": CatBoostClassifier(random_state=RANDOM_STATE, verbose=0),
            "params": {
                "classifier__iterations": [100, 200],
                "classifier__depth": [4, 6, 8],
                "classifier__learning_rate": [0.01, 0.05, 0.1]
            }
        }
        
    # Benchmarking dictionary
    cv_scores = {}
    best_candidate_pipelines = {}
    
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    
    # We will score based on F1 due to target class imbalance
    for name, config in models_to_tune.items():
        logger.info(f"Tuning hyperparameters for {name}...")
        
        # Use SMOTE inside pipeline to prevent target leakage
        pipeline = create_full_pipeline(preprocessor, config["model"], use_smote=True)
        
        # Hyperparameter search
        search = RandomizedSearchCV(
            estimator=pipeline,
            param_distributions=config["params"],
            n_iter=10,
            scoring="f1",
            cv=cv,
            random_state=RANDOM_STATE,
            n_jobs=-1
        )
        
        try:
            search.fit(X_train, y_train)
            best_candidate_pipelines[name] = search.best_estimator_
            cv_scores[name] = search.best_score_
            logger.info(f"{name} best cross-validation F1-Score: {search.best_score_:.4f}")
        except Exception as e:
            logger.error(f"Failed to fit {name} model: {e}")
            
    # 4. Compare Models on Test Set
    test_metrics = {}
    
    logger.info("Evaluating candidates on test set...")
    for name, pipeline in best_candidate_pipelines.items():
        metrics, _ = evaluate_model(pipeline, X_test, y_test, name, FIGURES_DIR)
        test_metrics[name] = metrics
        
    # 5. Selection: Best model selection based primarily on F1-score, fallback to ROC-AUC
    # We compute a score = 0.6 * F1 + 0.4 * ROC-AUC to balance both metrics
    best_model_name = None
    best_overall_score = -1.0
    
    for name, metrics in test_metrics.items():
        combined_score = 0.6 * metrics["f1_score"] + 0.4 * metrics["roc_auc"]
        logger.info(f"Model {name} - Test F1: {metrics['f1_score']:.4f}, ROC-AUC: {metrics['roc_auc']:.4f}, Score: {combined_score:.4f}")
        
        if combined_score > best_overall_score:
            best_overall_score = combined_score
            best_model_name = name
            
    logger.info(f"Selected Best Model: {best_model_name}")
    best_pipeline = best_candidate_pipelines[best_model_name]
    
    # 6. Pre-calculate details for interpretability (SHAP background)
    fitted_preprocessor = best_pipeline.named_steps["preprocessor"]
    X_train_proc = fitted_preprocessor.transform(X_train)
    feature_names = list(fitted_preprocessor.get_feature_names_out())
    
    # Save global SHAP summary
    logger.info("Calculating SHAP for global summary...")
    explainer, explainer_type = get_explainer(best_pipeline, X_train_proc)
    save_global_shap_summary(best_pipeline, explainer, X_train_proc, feature_names, FIGURES_DIR)
    
    # 7. Save model package
    model_package = {
        "model_name": best_model_name,
        "pipeline": best_pipeline,
        "feature_names": feature_names,
        "categorical_features": categorical_features,
        "numerical_features": numerical_features,
        "X_train_proc": X_train_proc,
        "X_train": X_train,
        "test_metrics": test_metrics,
        "best_metrics": test_metrics[best_model_name]
    }
    
    Path(MODEL_PATH).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model_package, MODEL_PATH)
    logger.info(f"Saved best model package to {MODEL_PATH}")
    
    # Save test metrics summary JSON
    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(test_metrics, f, indent=4)
        
    logger.info("Model training process complete.")
    
if __name__ == "__main__":
    train_and_select_best_model()
