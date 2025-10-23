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
import sys
import os
# Add the project root to the path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, 'src'))

from biastrack.data.preprocess_v1 import DataPreprocessor
from biastrack.data.data_loader import DataLoader
from biastrack.train.model_v1.trainer import Trainer

# -------------------------------
if __name__ == "__main__":

    # Step 1: Load data and Preprocess data
    def preprocess_data():
        data_loader = DataLoader()
        df = data_loader.load_csv('glassdoor_gender_pay_gap.csv')
        preprocessor_linear = DataPreprocessor()
        X, y = preprocessor_linear.fit_transform(df)
        preprocessor_linear.save_preprocessor()
        return X, y

    # Step 2: Train model
    X, y = preprocess_data()
    trainer = Trainer()
    model = trainer.train(X, y)

    # Step 3: Save Model
    trainer.save_model(model)

    print("✅ Training pipeline completed successfully!")
   