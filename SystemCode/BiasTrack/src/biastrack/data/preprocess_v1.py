# ===============================
# Data Preprocessing Module
# ===============================
# This module contains classes and functions responsible for:
# - Cleaning raw data
# - Handling missing values
# - Encoding categorical features
# - Scaling/normalizing numerical features
# - Feature engineering
# - Saving/loading preprocessing objects
# ===============================

# -------------------------------
# 1. Import Libraries
# -------------------------------
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder
import joblib
import os

# -------------------------------
# 2. Class: DataPreprocessor
# -------------------------------
# Purpose: Encapsulate all preprocessing logic for the dataset
# Responsibilities:
# - Fit and transform training data
# - Transform new/unseen data
# - Save/load preprocessing artifacts

class DataPreprocessor:
    
    # A class for preprocessing the bias tracking dataset.
    # Supports preprocessing for regression models (linear/logistic).
    # Assumes categorical features: JobTitle, Gender, Education, Dept
    # Numerical features: Age, PerfEval, Seniority
    # Target can be specified (e.g., 'BasePay' for linear regression, 'Gender' for logistic).
    # For logistic regression, target should be binary; preprocessing handles it similarly.
    

    def __init__(self,target_column = None):
        self.categorical_features = None  # To be set dynamically
        self.numerical_features = None  # To be set dynamically
        self.preprocessor = None
        self.feature_names = None  # To store transformed feature names
        self.target_column = target_column  # To be set dynamically

    def fit(self, df):

        # Fit the preprocessor on training data.
        # Handles imputation, encoding, and scaling.
        
        df['Total Pay']=df['BasePay']+df['Bonus']
        df.drop(columns=['BasePay','Bonus'],inplace=True)
        self.target_column = 'Total Pay'

        # Dynamically infer numerical and categorical features
        feature_columns = [col for col in df.columns if col != self.target_column]
        self.numerical_features = [col for col in feature_columns if df[col].dtype.kind in 'biufc']  # bool, int, uint, float, complex
        self.categorical_features = [col for col in feature_columns if col not in self.numerical_features]

        # Separate features and target
        X = df[self.numerical_features + self.categorical_features].copy()
        y = df[self.target_column]

        # Normalizing the features - for ['Age','Education'] we can use Ordinal Encoding , for ['JobTitle','Dept''Gender'] we can use OneHotEncoding
        onehot_features=[]
        ordinal_features=[]
        standard_scaler_cols=[]

        for col in X.columns:
            if col in ['Dept','JobTitle','Gender']:
                onehot_features.append(col)
            elif col in ['Education']:
                ordinal_features.append(col)
            else:
                standard_scaler_cols.append(col)

        combined_pipeline=ColumnTransformer(
            [
                ('onehot', OneHotEncoder(handle_unknown='ignore',drop='first'), onehot_features),
                ('ordinal', OrdinalEncoder(), ordinal_features),
                ('standard_scaler', StandardScaler(), standard_scaler_cols)
            ]
        )

        self.preprocessor=combined_pipeline.fit(X)

        return self.preprocessor.transform(X), y
    

    def transform(self, df):

        # Transform new data using the fitted preprocessor.
        # Returns transformed features and original target (or encoded if categorical).
        if 'BasePay' in df.columns and 'Bonus' in df.columns:
            df['Total Pay']=df['BasePay']+df['Bonus']
            df.drop(columns=['BasePay','Bonus'],inplace=True)
            self.target_column = 'Total Pay'

        if self.preprocessor is None:
            raise ValueError("Preprocessor not fitted. Call fit() first.")

        # Ensure required columns exist
        feature_cols = self.numerical_features + self.categorical_features
        missing_cols = [col for col in feature_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing columns in data: {missing_cols}")

        X = df[feature_cols].copy()
        y = df[self.target_column] if self.target_column in df.columns else None

        X_transformed = self.preprocessor.transform(X)

        return X_transformed, y

    def fit_transform(self,df):
        
        # Convenience method to fit and transform in one step.
        X_transformed, y = self.fit(df)
        return X_transformed, y
    
    def save_preprocessor(self, version='v1',path='artifacts/model_v1/preprocessors/preprocessor.pkl'):

        if path is None:
         path = f'artifacts/model_{version}/preprocessors/preprocessor.pkl'

        # Save the fitted preprocessor to disk

        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump({
            'preprocessor': self.preprocessor,
            'feature_names': self.feature_names,
            'target_column': self.target_column,
            'numerical_features': self.numerical_features,
            'categorical_features': self.categorical_features
        }, path)
        print(f"Preprocessor saved to {path}")

    @classmethod
    def load_preprocessor(cls, path='artifacts/model_v1/preprocessors/preprocessor.pkl'):
        
        # Load the preprocessor from disk.
        
        data = joblib.load(path)
        preprocessor = cls(target_column=data['target_column'])
        preprocessor.preprocessor = data['preprocessor']
        preprocessor.feature_names = data['feature_names']
        preprocessor.numerical_features = data['numerical_features']
        preprocessor.categorical_features = data['categorical_features']
        return preprocessor