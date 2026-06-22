import logging
import json
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, Any, Tuple
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    roc_auc_score, confusion_matrix, classification_report,
    roc_curve, precision_recall_curve
)
from sklearn.calibration import calibration_curve

logger = logging.getLogger(__name__)

def evaluate_model(
    model, 
    X_test: pd.DataFrame, 
    y_test: pd.Series, 
    model_name: str,
    figures_dir: Path
) -> Tuple[Dict[str, Any], np.ndarray]:
    """
    Evaluates a model and returns a metrics dictionary and confusion matrix.
    Generates and saves performance plots to the figures directory.
    """
    logger.info(f"Evaluating model: {model_name}")
    
    # Predictions
    y_pred = model.predict(X_test)
    
    # Handle probability predictions if supported
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test)[:, 1]
    else:
        # For classifiers without predict_proba (e.g. some SVM configurations)
        if hasattr(model, "decision_function"):
            scores = model.decision_function(X_test)
            y_prob = 1 / (1 + np.exp(-scores))  # Sigmoid mapping
        else:
            y_prob = y_pred.astype(float)
            
    # Metrics calculations
    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1_score": float(f1_score(y_test, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, y_prob))
    }
    
    logger.info(f"Metrics for {model_name}: {metrics}")
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    
    # Save plots
    figures_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Confusion Matrix Plot
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", 
                xticklabels=["No", "Yes"], yticklabels=["No", "Yes"])
    plt.title(f"Confusion Matrix - {model_name}")
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.tight_layout()
    cm_path = figures_dir / f"confusion_matrix_{model_name.lower().replace(' ', '_')}.png"
    plt.savefig(cm_path, dpi=300)
    plt.close()
    
    # 2. ROC & Precision-Recall Curves
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # ROC Curve
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    ax1.plot(fpr, tpr, label=f"ROC Curve (AUC = {metrics['roc_auc']:.3f})", color="darkorange", lw=2)
    ax1.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
    ax1.set_xlim([0.0, 1.0])
    ax1.set_ylim([0.0, 1.05])
    ax1.set_xlabel("False Positive Rate")
    ax1.set_ylabel("True Positive Rate")
    ax1.set_title(f"ROC Curve - {model_name}")
    ax1.legend(loc="lower right")
    ax1.grid(True, linestyle="--", alpha=0.7)
    
    # PR Curve
    precision, recall, _ = precision_recall_curve(y_test, y_prob)
    ax2.plot(recall, precision, label=f"PR Curve (F1 = {metrics['f1_score']:.3f})", color="green", lw=2)
    ax2.set_xlim([0.0, 1.0])
    ax2.set_ylim([0.0, 1.05])
    ax2.set_xlabel("Recall")
    ax2.set_ylabel("Precision")
    ax2.set_title(f"Precision-Recall Curve - {model_name}")
    ax2.legend(loc="lower left")
    ax2.grid(True, linestyle="--", alpha=0.7)
    
    plt.tight_layout()
    curves_path = figures_dir / f"curves_{model_name.lower().replace(' ', '_')}.png"
    plt.savefig(curves_path, dpi=300)
    plt.close()
    
    # 3. Calibration Curve
    prob_true, prob_pred = calibration_curve(y_test, y_prob, n_bins=10)
    plt.figure(figsize=(6, 5))
    plt.plot(prob_pred, prob_true, marker="o", linewidth=1, label=model_name)
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfect Calibration")
    plt.xlabel("Predicted probability")
    plt.ylabel("True probability")
    plt.title(f"Calibration Curve - {model_name}")
    plt.legend(loc="lower right")
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.tight_layout()
    cal_path = figures_dir / f"calibration_{model_name.lower().replace(' ', '_')}.png"
    plt.savefig(cal_path, dpi=300)
    plt.close()
    
    return metrics, cm
