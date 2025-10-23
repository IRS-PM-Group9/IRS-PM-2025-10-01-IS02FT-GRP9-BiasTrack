import sys
import os
sys.path.append('src')

from src.biastrack.data.preprocess import DataPreprocessor
from src.biastrack.data.data_loader import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

# Step 1: Load data and Preprocess data
data_loader = DataLoader()
df = data_loader.load_csv('glassdoor_gender_pay_gap.csv')
preprocessor_linear = DataPreprocessor()
X, y = preprocessor_linear.fit_transform(df)
preprocessor_linear.save_preprocessor()

# Step 2: Train model
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, random_state=123)
model = LinearRegression()
model.fit(X_train, y_train)

# predict the data
y_predict = model.predict(X_test)

# --- Evaluation ---
mae = mean_absolute_error(y_test, y_predict)
mse = mean_squared_error(y_test, y_predict)
r2 = r2_score(y_test, y_predict)

print(f"📊 MAE: {mae:.2f}")
print(f"📉 MSE: {mse:.2f}")
print(f"📈 R² Score: {r2:.2f}")

# --- Save Model ---
os.makedirs("artifacts/model_v1/models", exist_ok=True)
joblib.dump(model, "artifacts/model_v1/models/modelLinear")
print("✅ Model saved successfully!")
