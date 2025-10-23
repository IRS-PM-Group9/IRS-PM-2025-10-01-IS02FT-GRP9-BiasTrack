# ===============================
# Model Training Module
# ===============================
# This module contains classes and functions responsible for:
# - Splitting data into training and validation sets
# - Training regression models (Linear, Ridge)
# - Managing model initialization and fitting
# - Handling model persistence (saving/loading)
# ===============================

# -------------------------------
# 1. Import Libraries
# -------------------------------
import os, joblib, json
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import RidgeCV
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# -------------------------------
# 2. Class: ModelTrainer
# -------------------------------
# Purpose: Encapsulate the end-to-end training workflow for regression models.
# Responsibilities:
# - Accept preprocessed data to build a Pipeline
# - Initialize and fit Linear and Ridge regression models
# - Performs hyperparameter search (e.g., Ridge alpha grid)
# - Manage train/val split
# - Persist trained artifacts (models) and training metadata
# - Expose a simple interface to retrieve the best, inference-ready artifact

class ModelTrainer:

    def __init__(self):
        """Initialise model training objects"""

        self.best_alpha_ = None # set by ridge regression

        # set after model evaluation
        self.best_model_ = None
        self.best_metrics_ = None
        self.best_model_name_ = None

    # ---- Public Functions ---        
    def fit(self, X, y):
        """Train the model on the provided dataset and save the trained artifact"""

        # 1. Split dataset into training, validation and test (save test data for evaluation later)
        X_train, X_val, y_train, y_val = self._split_data(X, y)

        # 2. Train candidate models
        model_lr = self.train_linear_regression(X_train, y_train)
        model_rcv = self.train_ridge_regression_cv(X_train, y_train)

        # 3. Evaluate models on validation set
        metrics_lr = self._evaluate_model(model_lr, X_val, y_val)
        metrics_rcv = self._evaluate_model(model_rcv, X_val, y_val)

        # 4. Select better model
        self._select_better_model(metrics_lr, model_lr, metrics_rcv, model_rcv)

        return self.best_model_, self.best_metrics_, self.best_model_name_

    def save(self, model_path, metadata_path):
        """Save the trained model and its metadata to artifacts"""

        # 1. Ensure the target folders exist
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        os.makedirs(os.path.dirname(metadata_path), exist_ok=True)

        # 2. Save the trained best model
        joblib.dump(self.best_model_, model_path)

        # 3. Save metadata (model name, metrics, hyperparams)
        metadata = {
            "model_name": getattr(self, "best_model_name_", None),
            "metrics": getattr(self, "best_metrics_", None),
            "best_alpha": getattr(self, "best_alpha_", None),
        }

        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=4)

        print(f"Model saved to: {model_path}")
        print(f"Metadata saved to: {metadata_path}")

    @classmethod
    def load(cls, model_path):
        """Load a saved model into a new ModelTrainer instance"""

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at: {model_path}")

        instance = cls()
        instance.best_model_ = joblib.load(model_path)
        print(f"Model loaded successfully from: {model_path}")

        return instance
    
    # ---- Helper Functions ---
    def _split_data(self, X, y, test_size=0.15, val_size=0.15, random_state=42):
        """
        Split the dataset into training, validation and test sets.
        Saving test set for evaluation.
        """
        # 1. Splitting the test set
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )

        # 2. Save test indices
        os.makedirs("src/biastrack/data/splits", exist_ok=True)
        test_indices = X_test.index.to_numpy()
        np.save("src/biastrack/data/splits/test_indices.npy", test_indices)

        # 3. Split remaining data into train and validation sets
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp,
            test_size=val_size / (1 - test_size),  # proportion relative to remaining 85%
            random_state=random_state
        )

        # 4. Return all splits
        return X_train, X_val, y_train, y_val


    def train_linear_regression(self, X_train, y_train):
        """Train a Linear Regression model and return the fitted estimator"""
        model = LinearRegression()
        model.fit(X_train, y_train)
        return model

    # RidgeCV: automates both model training and hyperparameter tuning in one go
    # CV: K-Fold Cross-Validation
    # alpha: Regularization strength
    def train_ridge_regression_cv(self, X_train, y_train, alphas=(0.1, 0.3, 1.0, 3.0, 10.0), cv=5):
        """Train Ridge regression with cross-validation over alphas; return the fitted best estimator"""
       
        # choosing rmse as evaluation for CVs
        model = RidgeCV(alphas=alphas, cv=cv, scoring="neg_root_mean_squared_error")
        model.fit(X_train, y_train)

        # Keep track of the chosen alpha
        self.best_alpha_ = getattr(model, "alpha_", None)

        return model

    def _evaluate_model(self, model, X_val, y_val):
        """Return validation metrics for a fitted model."""

        y_pred = model.predict(X_val)

        # How much variation model explains (1 = perfect, 0 = useless)
        r2 = r2_score(y_val, y_pred)

        # Typical size of error, Lower = better
        rmse = np.sqrt(mean_squared_error(y_val, y_pred))

        # Average absolute error (Lower = better)
        mae = mean_absolute_error(y_val, y_pred)

        return {"r2": r2, "rmse": rmse, "mae": mae}
    
    def _select_better_model(self, metrics_lr, model_lr, metrics_rcv, model_rcv):

        if (
            (metrics_rcv["rmse"] <= metrics_lr["rmse"]) and
            (metrics_rcv["mae"] <= metrics_lr["mae"]) and
            (metrics_rcv["r2"] >= metrics_lr["r2"])
        ):
            self.best_model_ = model_rcv
            self.best_metrics_ = metrics_rcv
            self.best_model_name_ = "ridge_cv"
        else:
            self.best_model_ = model_lr
            self.best_metrics_ = metrics_lr
            self.best_model_name_ = "linear_regression"

    