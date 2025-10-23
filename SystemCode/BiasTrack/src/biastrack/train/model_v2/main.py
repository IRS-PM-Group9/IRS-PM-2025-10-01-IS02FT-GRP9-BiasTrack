# ===============================
# Main Orchestrator Script
# ===============================
# This script is responsible for:
# - Coordinating the end-to-end workflow of the project
# - Loading raw data using DataLoader
# - Preprocessing data using DataPreprocessor
# - Training and evaluating models
# - Saving artifacts (models, preprocessors, metadata)
# ===============================

# -------------------------------
# 1. Import Libraries
# -------------------------------
from src.biastrack.data.data_loader import DataLoader
from src.biastrack.data.preprocess_v2 import DataPreprocessor
from src.biastrack.train.model_v2.trainer import ModelTrainer
from src.biastrack.train.model_v2.evaluate import ModelEvaluator
import numpy as np
from pathlib import Path

def main():

    # Defining Project Paths
    ARTIFACTS_DIR = Path("artifacts") / "model_v2"
    MODELS_DIR = ARTIFACTS_DIR / "models"
    PREPROCESSORS_DIR = ARTIFACTS_DIR / "preprocessors"
    METADATA_DIR = ARTIFACTS_DIR / "metadata"

    # -------------------------------
    # Step 1: Load data
    # -------------------------------
    loader = DataLoader()
    df = loader.load_csv("glassdoor_gender_pay_gap.csv")

    # -------------------------------
    # Step 2: Preprocess data
    # -------------------------------
    preprocessor = DataPreprocessor()
    X_features, y = preprocessor.fit_transform(df) # fit + transform training data

    # -------------------------------
    # Step 3: Train model
    # -------------------------------
    model = ModelTrainer()
    model.fit(X_features, y)

    # -------------------------------
    # Step 4: Save Model and Preprocessor
    # -------------------------------
    preprocessor.save(PREPROCESSORS_DIR / "preprocessor.joblib")
    model.save(
        MODELS_DIR / "model.joblib",
        METADATA_DIR / "metadata.json"
    )

    # -------------------------------
    # Step 5: Evaluate Model
    # -------------------------------

    # Load the saved model to pass to evaluation on test data
    saved_model = model.load(MODELS_DIR / "model.joblib")

    # Load the saved test indices
    test_indices = np.load(Path("src/biastrack/data/splits") / "test_indices.npy")

    evaluator = ModelEvaluator()
    evaluator.evaluate(saved_model, X_features, y, test_indices)


# -------------------------------
# Entry Point
# -------------------------------
if __name__ == "__main__":
    main()