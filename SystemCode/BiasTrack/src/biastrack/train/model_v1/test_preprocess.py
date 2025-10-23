from src.biastrack.data.data_loader import DataLoader
from src.biastrack.data.preprocess import DataPreprocessor

# Load data
loader = DataLoader()
df = loader.load_csv('glassdoor_gender_pay_gap.csv')
print("Data loaded. Shape:", df.shape)
print("Columns:", df.columns.tolist())
print("Sample data:")
print(df.head())

# Preprocess for linear regression (BasePay)
print("\n--- Linear Regression Preprocessing ---")
preprocessor_linear = DataPreprocessor(target_column='BasePay')
X_linear, y_linear = preprocessor_linear.fit(df)
X_linear, y_linear = preprocessor_linear.transform(df)
print("X_linear shape:", X_linear.shape)
print("y_linear shape:", y_linear.shape)
print("Feature names:", preprocessor_linear.feature_names[:5])  # First 5

# Preprocess for logistic regression (Gender)
print("\n--- Logistic Regression Preprocessing ---")
preprocessor_logistic = DataPreprocessor(target_column='Gender')
X_logistic, y_logistic = preprocessor_logistic.fit(df)
X_logistic, y_logistic = preprocessor_logistic.transform(df)
print("X_logistic shape:", X_logistic.shape)
print("y_logistic shape:", y_logistic.shape)
print("Unique y_logistic:", set(y_logistic))


