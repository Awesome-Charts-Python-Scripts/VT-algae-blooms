import numpy as np
import sklearn
import argparse
from pprint import pprint
from datetime import date
from typing import Tuple, List, Dict
import matplotlib.pyplot as plt


def generate_roc_curve_plot(model_name: str, y: np.array, y_pred: np.array, dst: str):
    cv_auc = sklearn.metrics.roc_auc_score(y, y_pred)
    false_positive_rate, true_positive_rate, _ = sklearn.metrics.roc_curve(y, y_pred)
    plt.figure(figsize=(6, 6))
    plt.plot(false_positive_rate, true_positive_rate, label=f"ROC (AUC = {cv_auc:.3f})")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"{model_name} ROC Curve (Out-of-Fold)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(dst, dpi=300, bbox_inches="tight")


def generate_precision_recall_curve_plot(
    model_name: str, y: np.array, y_pred: np.array, dst: str
):
    precision, recall, _ = sklearn.metrics.precision_recall_curve(y, y_pred)
    average_precision = sklearn.metrics.average_precision_score(y, y_pred)
    plt.figure(figsize=(6, 6))
    plt.plot(recall, precision, label=f"Average Precision = {average_precision:.3f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"{model_name} Precision–Recall Curve")
    plt.legend()
    plt.grid(True)
    x_min, x_max = plt.xlim()
    y_min, y_max = plt.ylim()
    x_pad = x_max - 1
    y_pad = y_max - 1
    plt.xlim(-x_pad, 1 + x_pad)
    plt.ylim(-y_pad, 1 + y_pad)
    plt.savefig(dst, dpi=300, bbox_inches="tight")
