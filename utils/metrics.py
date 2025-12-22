#!/usr/bin/env python3
"""
Metrics Utilities for Medical Image Classification
Custom metrics for evaluating medical image classification models.
"""

import numpy as np
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_recall_fscore_support,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    average_precision_score,
)
from typing import Dict, Tuple, List


def calculate_sensitivity_specificity(
    y_true: np.ndarray, y_pred: np.ndarray
) -> Tuple[float, float]:
    """Calculate sensitivity and specificity."""
    cm = confusion_matrix(y_true, y_pred)

    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    else:
        sensitivity = 0.0
        specificity = 0.0

    return sensitivity, specificity


def calculate_ppv_npv(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[float, float]:
    """Calculate Positive Predictive Value and Negative Predictive Value."""
    cm = confusion_matrix(y_true, y_pred)

    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        ppv = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        npv = tn / (tn + fn) if (tn + fn) > 0 else 0.0
    else:
        ppv = 0.0
        npv = 0.0

    return ppv, npv


def calculate_likelihood_ratios(
    sensitivity: float, specificity: float
) -> Tuple[float, float]:
    """Calculate positive and negative likelihood ratios."""
    fpr = 1 - specificity
    fnr = 1 - sensitivity

    positive_lr = sensitivity / fpr if fpr > 0 else float("inf")
    negative_lr = fnr / specificity if specificity > 0 else 0.0

    return positive_lr, negative_lr


def calculate_clinical_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray
) -> Dict[str, float]:
    """Calculate comprehensive clinical metrics."""
    # Basic metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0
    )

    # AUC metrics
    auc_roc = roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else 0.0
    auc_pr = (
        average_precision_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else 0.0
    )

    # Medical-specific metrics
    sensitivity, specificity = calculate_sensitivity_specificity(y_true, y_pred)
    ppv, npv = calculate_ppv_npv(y_true, y_pred)
    positive_lr, negative_lr = calculate_likelihood_ratios(sensitivity, specificity)

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "auc_roc": auc_roc,
        "auc_pr": auc_pr,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "ppv": ppv,
        "npv": npv,
        "positive_likelihood_ratio": positive_lr,
        "negative_likelihood_ratio": negative_lr,
    }


def find_optimal_threshold(
    y_true: np.ndarray, y_prob: np.ndarray, metric: str = "youden"
) -> Tuple[float, Dict[str, float]]:
    """Find optimal classification threshold based on specified metric."""
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)

    if metric == "youden":
        # Youden's J statistic = Sensitivity + Specificity - 1
        j_scores = tpr - fpr
        optimal_idx = np.argmax(j_scores)
    elif metric == "f1":
        # Maximize F1 score
        f1_scores = []
        for threshold in thresholds:
            y_pred = (y_prob >= threshold).astype(int)
            _, _, f1, _ = precision_recall_fscore_support(
                y_true, y_pred, average="binary", zero_division=0
            )
            f1_scores.append(f1)
        optimal_idx = np.argmax(f1_scores)
    else:
        optimal_idx = np.argmax(tpr - fpr)

    optimal_threshold = thresholds[optimal_idx]

    # Calculate metrics at optimal threshold
    y_pred_optimal = (y_prob >= optimal_threshold).astype(int)
    metrics_at_optimal = calculate_clinical_metrics(y_true, y_pred_optimal, y_prob)

    return optimal_threshold, metrics_at_optimal


def calculate_confidence_intervals(
    metric_values: List[float], confidence_level: float = 0.95
) -> Tuple[float, float]:
    """Calculate confidence intervals using bootstrap."""
    from scipy import stats

    mean = np.mean(metric_values)
    stderr = stats.sem(metric_values)
    interval = stderr * stats.t.ppf(
        (1 + confidence_level) / 2.0, len(metric_values) - 1
    )

    return mean - interval, mean + interval


def evaluate_model_calibration(
    y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10
) -> Dict[str, any]:
    """Evaluate model calibration."""
    from sklearn.calibration import calibration_curve

    fraction_of_positives, mean_predicted_value = calibration_curve(
        y_true, y_prob, n_bins=n_bins
    )

    # Expected Calibration Error (ECE)
    ece = np.mean(np.abs(fraction_of_positives - mean_predicted_value))

    return {
        "expected_calibration_error": ece,
        "fraction_of_positives": fraction_of_positives,
        "mean_predicted_value": mean_predicted_value,
    }
