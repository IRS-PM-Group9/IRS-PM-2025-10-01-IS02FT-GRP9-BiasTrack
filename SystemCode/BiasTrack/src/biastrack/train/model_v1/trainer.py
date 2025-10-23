# ===============================
# Trainer Module
# ===============================
# This module contains the Trainer class responsible for:
# - Training machine learning models
# - Evaluating model performance
# - Saving trained models to disk
# ===============================
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

class Trainer:
    def __init__(self, model_dir='artifacts/model_v1/models', test_size=0.3, random_state=123):
        self.model_dir = model_dir
        self.test_size = test_size
        self.random_state = random_state
        os.makedirs(self.model_dir, exist_ok=True)

    def train(self, X, y, model_type='linear'):
        """
        Train a model on the provided data.
        Args:
            X: Feature matrix
            y: Target vector
            model_type: Type of model to train ('linear' for now)
        Returns:
            Trained model
        """
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state
        )

        # Create and train model
        if model_type == 'linear':
            model = LinearRegression()
        else:
            raise ValueError(f"Unsupported model type: {model_type}")

        model.fit(X_train, y_train)

        # Evaluate
        y_predict = model.predict(X_test)
        mae = mean_absolute_error(y_test, y_predict)
        mse = mean_squared_error(y_test, y_predict)
        r2 = r2_score(y_test, y_predict)

        print("📊 Model Evaluation:")
        print(f"   MAE: {mae:.2f}")
        print(f"   MSE: {mse:.2f}")
        print(f"   R² Score: {r2:.2f}")

        return model

    def save_model(self, model, model_name='modelLinear'):
        """
        Save the trained model to disk.
        Args:
            model: Trained model object
            model_name: Name for the saved model file
        """
        model_path = os.path.join(self.model_dir, model_name)
        joblib.dump(model, model_path)
        print(f"✅ Model saved to {model_path}")
