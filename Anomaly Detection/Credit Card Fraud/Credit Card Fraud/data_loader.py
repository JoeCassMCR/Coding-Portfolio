"""
data_loader.py

Functions to load and validate the credit card fraud dataset.
"""
import os
import logging
import zipfile
import pandas as pd
from pandera.pandas import DataFrameSchema, Column, Check
from config import DATA_PATH, SCHEMA, DATA_DIR, ZIP_PATH

LOGGER = logging.getLogger(__name__)

def ensure_dataset():
    """
    If the CSV isn't present at DATA_PATH, unzip ZIP_PATH into DATA_DIR.
    """
    # 1) If we've already extracted, do nothing
    if os.path.exists(DATA_PATH):
        return

    # 2) ZIP must exist
    if not os.path.exists(ZIP_PATH):
        raise FileNotFoundError(f"Expected zip at {ZIP_PATH}, but none found.")

    # 3) Make the data folder and extract
    os.makedirs(DATA_DIR, exist_ok=True)
    LOGGER.info(f"Extracting dataset from {ZIP_PATH} to {DATA_DIR} …")
    with zipfile.ZipFile(ZIP_PATH, 'r') as z:
        z.extractall(DATA_DIR)
    LOGGER.info("Extraction complete.")


    
def load_data(csv_path: str = DATA_PATH) -> pd.DataFrame:
    """
    Load the Kaggle Credit Card Fraud dataset from a CSV and validate schema.

    Raises error if data does not conform.
    """
    LOGGER.info(f"Loading data from {csv_path} ...")
    ensure_dataset()
    df = pd.read_csv(csv_path)
    LOGGER.info(f"Dataset shape: {df.shape}")

    # Validate schema using Pandera
    schema = DataFrameSchema(
        columns={
            "Time": Column(float),
            **{f"V{i}": Column(float) for i in range(1, 29)},
            "Amount": Column(float),
            "Class": Column(int, checks=Check.isin([0, 1])),
        }
    )
    df = schema.validate(df)
    return df
