import logging
import shap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Any, Tuple

logger = logging.getLogger(__name__)

def get_explainer(pipeline, X_train_proc: np.ndarray) -> Tuple[Any, str]:
    """
    Creates the appropriate SHAP explainer based on the classifier type.
    """
    classifier = pipeline.named_steps["classifier"]
    classifier_name = classifier.__class__.__name__
    
    logger.info(f"Creating SHAP explainer for model type: {classifier_name}")
    
    # Select appropriate explainer
    if classifier_name in ["RandomForestClassifier", "XGBClassifier", "LGBMClassifier", "CatBoostClassifier", "GradientBoostingClassifier"]:
        # Tree-based explainer
        explainer = shap.TreeExplainer(classifier, X_train_proc)
        explainer_type = "tree"
    elif classifier_name in ["LogisticRegression"]:
        # Linear explainer
        explainer = shap.LinearExplainer(classifier, X_train_proc)
        explainer_type = "linear"
    else:
        # Fallback kernel/permutation explainer
        # We sample training data to speed up kernel SHAP
        background = shap.kmeans(X_train_proc, 10) if len(X_train_proc) > 10 else X_train_proc
        explainer = shap.KernelExplainer(classifier.predict_proba, background)
        explainer_type = "kernel"
        
    return explainer, explainer_type

def calculate_shap_values(explainer, x_proc: np.ndarray) -> Tuple[np.ndarray, float]:
    """
    Computes SHAP values and safe-extracts the scores for the positive class (1).
    """
    # Compute SHAP values
    if hasattr(explainer, "shap_values"):
        # Older api / kernel explainer fallback
        shap_out = explainer.shap_values(x_proc)
    else:
        shap_out = explainer(x_proc)
        
    # Standardize output format
    if isinstance(shap_out, list):
        # List per class (usually index 1 is class 1)
        shap_vals = shap_out[1]
        base_val = explainer.expected_value[1] if hasattr(explainer, "expected_value") else 0.5
    elif hasattr(shap_out, "values"):
        # shap.Explanation object
        vals = shap_out.values
        base_val = shap_out.base_values
        
        if len(vals.shape) == 3:
            # (samples, features, classes)
            shap_vals = vals[:, :, 1]
            if isinstance(base_val, np.ndarray):
                base_val = base_val[0, 1] if len(base_val.shape) == 2 else base_val[1]
        else:
            shap_vals = vals
            if isinstance(base_val, np.ndarray) and len(base_val.shape) > 0:
                base_val = base_val[0]
    else:
        # For tree/linear models with binary output, SHAP values are sometimes direct arrays
        shap_vals = shap_out
        base_val = explainer.expected_value
        if isinstance(base_val, (list, np.ndarray)):
            base_val = base_val[1] if len(base_val) > 1 else base_val[0]
            
    # If the shape is 2D and it represents a single sample, squeeze it
    return np.array(shap_vals), float(base_val)

def get_individual_explanation(
    pipeline, 
    explainer, 
    x_input: pd.DataFrame
) -> Dict[str, Any]:
    """
    Generates a localized explanation dictionary for a single employee record.
    """
    preprocessor = pipeline.named_steps["preprocessor"]
    feature_names = list(preprocessor.get_feature_names_out())
    
    # Process input
    x_proc = preprocessor.transform(x_input)
    
    # Calculate SHAP values
    shap_vals, base_val = calculate_shap_values(explainer, x_proc)
    
    # Squeeze to 1D for single prediction analysis
    if len(shap_vals.shape) > 1:
        shap_vals = shap_vals[0]
        
    # Map feature names to SHAP values
    explanation = {
        "base_value": base_val,
        "prediction_probability": float(pipeline.predict_proba(x_input)[0, 1]),
        "feature_explanations": [
            {
                "feature": name,
                "display_name": name.replace("num__", "").replace("cat__", ""),
                "shap_value": float(shap_vals[i]),
                "raw_value": x_input[name.split("__")[1]].iloc[0] if name.split("__")[1] in x_input.columns else "N/A"
            }
            for i, name in enumerate(feature_names)
        ]
    }
    
    # Sort feature explanations by impact size (absolute SHAP value)
    explanation["feature_explanations"] = sorted(
        explanation["feature_explanations"], 
        key=lambda x: abs(x["shap_value"]), 
        reverse=True
    )
    
    return explanation

def save_global_shap_summary(
    pipeline,
    explainer,
    X_train_proc: np.ndarray,
    feature_names: List[str],
    figures_dir: Path
) -> None:
    """
    Generates and saves a SHAP summary plot representing global feature importances.
    """
    logger.info("Generating and saving global SHAP summary plot.")
    
    # Compute SHAP values for training set
    shap_vals, _ = calculate_shap_values(explainer, X_train_proc)
    
    plt.figure(figsize=(10, 8))
    
    # Clean feature names for plot (remove pipeline prefixes)
    clean_feature_names = [f.replace("num__", "").replace("cat__", "") for f in feature_names]
    
    # Draw summary plot
    shap.summary_plot(
        shap_vals, 
        X_train_proc, 
        feature_names=clean_feature_names, 
        show=False
    )
    
    plt.title("SHAP Global Feature Importance (Impact on Attrition)", fontsize=14, pad=15)
    plt.tight_layout()
    
    save_path = figures_dir / "shap_global_summary.png"
    figures_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    
    logger.info(f"Global SHAP summary saved to {save_path}")
