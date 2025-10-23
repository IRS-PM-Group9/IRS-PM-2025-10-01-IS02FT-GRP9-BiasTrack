# ===============================
# Data Preprocessing Module
# ===============================
# This module contains classes and functions responsible for:
# - Cleaning raw data
# - Handling missing values
# - Feature engineering
# - Encoding categorical features
# - Scaling/normalizing numerical features
# - Finalizing Features
# - Saving/loading preprocessing objects
# ===============================

# -------------------------------
# 1. Import Libraries
# -------------------------------
import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler

# -------------------------------
# 2. Class: DataPreprocessor
# -------------------------------
# Purpose: Encapsulate all preprocessing logic for the dataset
# Responsibilities:
# - Fit and transform training data
# - Transform new/unseen data
# - Save/load preprocessing artifacts

class DataPreprocessor:

    # Initialize any preprocessing objects (scalers, encoders, etc.) to be set after fitting
    def __init__(self):
        self.numerical_features = None  # To be set dynamically
        self.categorical_features = None  # To be set dynamically
        
        self.feature_columns = None # To be set dynamically

        # Encoders/Scalers   
        self.ordinal_category_encoder = None # set after encoding (Ordinal-LabelEncoder)
        self.ordinal_category_encoder_source_col = None # set after encoding
        self.ordinal_category_encoder_encoded_col = None # set after encoding
        
        self.ohe_category_encoder = None # set after encoding (Nominal-OneHotEncoder)
        self.ohe_category_encoder_cols = None # set after encoding (column structure)
        
        self.numerical_scaler = None # set after column scaling (StandardScaler)
        self.numerical_scaler_scale_col = None # set after scaling
        self.numerical_scaler_scaled_col = None # set after scaling

    # -------- public API --------
    def fit(self, df):
        """Learn any parameters (e.g., means, encoders, scalers)."""

        # 1. Clean Raw Data
        df = self._clean_raw_data(df)

        # 2. Handle Missing Values
        df = self._handle_missing_values(df)

        # 3. Feature engineering (creates totalpay)
        df = self._feature_engineering(df)

        # 4. Fit encoders (creates education_encoded and one-hot columns)
        df = self._encode_categorical_features(df)

        # 5. Fit scaler (creates age_scaled)
        df = self._scale_numerical_features(df)

        # 6. Finalise Features
        self._finalize_feature_schema(
            df,
            exclude={"totalpay","basepay","bonus","jobtitle","gender","dept","education","age"}
        )

        return self

    def transform(self, df):
        """Apply preprocessing transformations."""
        df = df.copy()

        # 1–3: same deterministic preprocessing
        df = self._clean_raw_data(df)
        df = self._handle_missing_values(df)
        df = self._feature_engineering(df)

        # 4a. Ordinal transform (use stored LabelEncoder)
        if self.ordinal_category_encoder and self.ordinal_category_encoder_encoded_col:
            src = self.ordinal_category_encoder_source_col
            dst = self.ordinal_category_encoder_encoded_col or f"{src}_encoded"
            if src in df.columns:
                df[dst] = self.ordinal_category_encoder.transform(df[src].astype(str))

        # 4b. Nominal transform (use stored OneHotEncoder)
        if self.ohe_category_encoder is not None:
            # Ensure required nominal input columns exist (empty string if missing)
            required_nominals = list(self.ohe_category_encoder.feature_names_in_)
            for c in required_nominals:
                if c not in df.columns:
                    df[c] = ""

            encoded = self.ohe_category_encoder.transform(df[required_nominals])
            # Use the *same* column names/ordering as in fit
            encoded_df = pd.DataFrame(
                encoded,
                columns=self.ohe_category_encoder_cols,
                index=df.index,
            )
            df = pd.concat([df, encoded_df], axis=1)

        # 5. Numerical scaling (use stored StandardScaler(s))
        if self.numerical_scaler is not None:
            if self.numerical_scaler_scale_col in df.columns:
                df[[self.numerical_scaler_scaled_col]] = self.numerical_scaler.transform(df[[self.numerical_scaler_scale_col]])

        # 6. Keep only the locked feature set from fit()
        if self.feature_columns:
            # Fill any missing locked features with 0 to be robust
            for c in self.feature_columns:
                if c not in df.columns:
                    df[c] = 0
            df = df[self.feature_columns]

        return df

    def fit_transform(self, df):
        """Fit the preprocessor on the given dataframe and return the transformed features and target"""
        df = df.copy()

        # Separating target - Repeated Steps since the target is engineered
        df = self._clean_raw_data(df)
        df = self._handle_missing_values(df)
        df = self._feature_engineering(df)
        y = df["totalpay"]

        # Fitting
        self.fit(df)

        # Transforming and creating feature set
        X = self.transform(df)
        
        return X, y

    def save(self, path):
        """Saving the preprocesor in artifacts for use later"""
        joblib.dump(self, path)

    @classmethod
    def load_preprocessor(cls, path):
        """Loading the preprocesor from artifacts for use on new dataset"""
        return joblib.load(path)
    


    # --- Internal helpers --- (function names start with _)
    def _clean_raw_data(self, df):
        
        df = df.copy()
        
        # 1. Normalize column names: All lowercase and spaces trimmed (eg. JobTitle -> jobtitle)
        df.columns = [c.strip().lower() for c in df.columns]
        
        # 2. Drop irrelevant columns: Dropping empty columns
        drop_cols = [c for c in df.columns if "unnamed" in c]
        # -- inplace=True -> Apply this change directly to the existing DataFrame instead of returning a new modified copy
        df.drop(columns=drop_cols, inplace=True, errors="ignore")
        
        # 3. Remove duplicates enteries (rows)
        df.drop_duplicates(inplace=True)

        # 4. Convert all placeholder missing values to NaN
        missing_indicators = ["", " ", "na", "NA", "n/a", "none", "null", "-", "?"]
        df.replace(missing_indicators, np.nan, inplace=True)
        
        return df
    
    def _handle_missing_values(self, df):
        
        df = df.copy()

        # 1. Detect Columns
        # - Detect existing numeric columns (int or float) dynamically
        self.numerical_features = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
        # - Detect categorical columns dynamically
        self.categorical_features = df.select_dtypes(include=["object", "category"]).columns.tolist()

        # 2. Handle missing values as per industry-standard rule of thumb i.e.
        # - Median for numerical features
        for col in self.numerical_features:
            if df[col].isnull().any():
                median_value = df[col].median()
                df[col].fillna(median_value, inplace=True)
        # - Mode for categorical features
        for col in self.categorical_features:
            if df[col].isnull().any():
                mode_value = df[col].mode(dropna=True)
                # mode() returns a Series — take the first value if it exists
                if not mode_value.empty:
                    df[col].fillna(mode_value[0], inplace=True)
        return df
    
    def _feature_engineering(self, df, base_col="basepay", bonus_col="bonus", target_col="totalpay"):
        
        df = df.copy()

        # Create Required Feature
        if all(c in df.columns for c in [base_col, bonus_col]):
            df[target_col] = df[base_col] + df[bonus_col]

        return df
    
    def _encode_categorical_features(
        self,
        df,
        ordinal_col: str = "education",
        encoded_col: str = "education_encoded",
        nominal_cols: list = ["jobtitle", "gender", "dept"]
    ):
        """Encodes categorical features (nominal → one-hot, ordinal → label)."""
        df = df.copy()

        # --- 1. Ordinal encoding: education ---
        if ordinal_col in df.columns:
            label_encoder = LabelEncoder()
            # LabelEncoder doesn't create new columns automatically
            # Hence, creating a new column manually to preserve the original
            df[encoded_col] = label_encoder.fit_transform(df[ordinal_col].astype(str))
            # store for transform() use later
            self.ordinal_category_encoder = label_encoder
            self.ordinal_category_encoder_source_col = ordinal_col
            self.ordinal_category_encoder_encoded_col = encoded_col


        # --- 2. One-hot encoding: jobtitle, gender, dept ---
        # OneHotEncoder creates new columns w/o replacing the orignal (preserved)
        valid_nominals = [c for c in nominal_cols if c in df.columns]
        if valid_nominals:
            ohe = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
            encoded = ohe.fit_transform(df[valid_nominals])
            encoded_df = pd.DataFrame(
                encoded,
                columns=ohe.get_feature_names_out(valid_nominals), # eg. gender_Female, gender_Male, dept_HR
                index=df.index # same row alignment as the original
            )

            # Adding onehot encoded cols to original df
            df = pd.concat([df, encoded_df], axis=1)

            # Store encoder for later use 
            self.ohe_category_encoder = ohe
            # Store column order structure 
            self.ohe_category_encoder_cols = encoded_df.columns.tolist()

        return df

    def _scale_numerical_features(self, df, scale_col: str = "age", scaled_col: str = "age_scaled"):
        """
        Scaling column 'age'. 
        Other numerical cols like 'perfeval' and 'seniority' are ordinal. 
        column 'totalpay' is the target
        columns 'basepay' and 'bonus' won't be features
        """
        df = df.copy()

        if scale_col in df.columns:
            scaler = StandardScaler()
            df[[scaled_col]] = scaler.fit_transform(df[[scale_col]])

            # Store scaler for later use 
            self.numerical_scaler = scaler
            self.numerical_scaler_scale_col = scale_col
            self.numerical_scaler_scaled_col = scaled_col

        return df
    
    def _finalize_feature_schema(self, df, exclude=None):
        exclude = set(exclude or [])
        features = [c for c in df.columns if c not in exclude]
        self.feature_columns = features


            
