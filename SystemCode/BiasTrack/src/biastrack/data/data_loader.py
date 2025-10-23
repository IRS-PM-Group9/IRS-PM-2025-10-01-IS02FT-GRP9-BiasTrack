# ===============================
# Data Loading Module
# ===============================
# This module is responsible for:
# - Reading raw data from CSV
# - Returning data as a pandas DataFrame for downstream processing
# - (Future scope) Supporting multiple data sources and formats (API, Database)
# - (Future scope) Validating file paths and data formats
# - (Future scope) Handling common I/O errors gracefully
# ===============================

# -------------------------------
# 1. Import Libraries
# -------------------------------
import pandas as pd
from pathlib import Path

# -------------------------------
# 2. Class: DataLoader
# -------------------------------
class DataLoader:

    # Initialize DataLoader instance and set base path for CSV files
    def __init__(self, base_path: str = None):
        # Resolve project root dynamically
        project_root = Path(__file__).resolve().parents[3]  # root level
        self.base_path = Path(base_path) if base_path else project_root / "dataset/training"

    # Load a CSV file from the base path and return it as a pandas DataFrame.
    # **kwargs: Additional keyword arguments passed to pd.read_csv like header, nrows etc

    def load_csv(self, filename: str, **kwargs) -> pd.DataFrame:
    
        # Construct full path to the CSV file
        file_path = self.base_path / filename

        # Check if the file exists, raise error if not
        if not file_path.exists():
            raise FileNotFoundError(f"CSV file not found: {file_path}")

        # Read CSV into a DataFrame, passing any additional pandas arguments
        df = pd.read_csv(file_path, **kwargs)
        return df
