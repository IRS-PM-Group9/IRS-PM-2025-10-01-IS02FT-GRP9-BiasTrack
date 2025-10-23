# ===============================
# Model Evaluation Module
# ===============================
# This module contains classes and functions responsible for:
# - Loading saved trained model artifacts
# - Restoring the held-out test split (no data leakage)
# - Computing regression metrics (R^2, RMSE, MAE)
# - Generating diagnostics (residuals, error distribution)
# - Persisting an evaluation report under artifacts
# ===============================

# -------------------------------
# 1. Import Libraries
# -------------------------------
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import numpy as np
import pandas as pd
import json
import matplotlib.pyplot as plt

# -------------------------------
# 2. Class: ModelEvaluator
# -------------------------------
# Purpose: Encapsulate the model evaluation workflow for regression models.
# Responsibilities:
# - Load trained model and metadata artifacts
# - Reconstruct the held-out test split using saved indices
# - Generate predictions and compute core regression metrics (R², RMSE, MAE)
# - Save evaluation results and summary report under artifacts
# - Provide a clean interface for integration with main.py or future APIs

class ModelEvaluator:

    def __init__(self):
        pass

    # --- public functions
    def evaluate(self, saved_model, X, y, test_indices):

        # 1. Recreate Test Data
        X_test, y_test = self._recreate_test_data(X, y, test_indices)

        # 2. Make Predictions using saved model
        y_pred = saved_model.best_model_.predict(X_test)

        # 3. Compute metrics
        metrics = self._compute_metrics(y_test, y_pred)

        # 4. Save evaluation report and visualizations
        self._save_report(metrics, "artifacts/model_v2/evaluation/report.json")

        # 5. Create and Save Visualization Plots
        self._create_save_visualizations(y_test, y_pred, "artifacts/model_v2/evaluation")


    # --- helper functions ---
    def _recreate_test_data(self, X, y, test_indices):
        """Recreate the held-out test split using saved indices"""

        # convert test_indices to an integer array
        idx = np.asarray(test_indices, dtype=int)

        if isinstance(X, pd.DataFrame):
            X_test = X.iloc[idx]
        else:  # NumPy array
            X_test = X[idx]

        if isinstance(y, (pd.Series, pd.DataFrame)):
            y_test = y.iloc[idx]
        else:
            y_test = y[idx]

        return X_test, y_test

    def _compute_metrics(self, y_test, y_pred):
        """Compute core regression metrics on test data"""
        
        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))

        return {
            "r2": r2,
            "mae": mae,
            "rmse": rmse
        }
    
    def _save_report(self, metrics, output_path):
        """Save evaluation metrics (and any extras) to a JSON file"""

        with open(output_path, "w") as f:
            json.dump(metrics, f, indent=4)

        print(f"✅ Evaluation report saved at: {output_path}")

    def _create_save_visualizations(self, y_test, y_pred, save_dir):
        """Generate and save basic evaluation plots: predicted vs actual and residual distribution"""
        # 1. Predicted vs Actual
        plt.figure(figsize=(6, 6))
        plt.scatter(y_test, y_pred, alpha=0.6) # 60% opaque
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')  # ideal line
        plt.xlabel("Actual Values")
        plt.ylabel("Predicted Values")
        plt.title("Predicted vs Actual")
        plt.tight_layout()
        plt.savefig(f"{save_dir}/predicted_vs_actual.png")
        plt.close()

        # 2. Residual Distribution
        residuals = y_test - y_pred
        plt.figure(figsize=(6, 4))
        plt.hist(residuals, bins=25, alpha=0.7)
        plt.xlabel("Residuals (Actual - Predicted)")
        plt.ylabel("Frequency")
        plt.title("Residual Distribution")
        plt.tight_layout()
        plt.savefig(f"{save_dir}/residuals_distribution.png")
        plt.close()