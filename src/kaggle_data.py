# src/kaggle_data.py
"""
Kaggle Dataset Loader for Mental Disorder Prediction
- Downloads dataset using kagglehub
- Returns a Pandas DataFrame
"""

import pandas as pd
import kagglehub

def load_mental_disorders_dataset(dataset_name="mdsultanulislamovi/mental-disorders-dataset"):
    """
    Downloads and loads the Mental Disorders dataset from Kaggle.

    Args:
        dataset_name (str): Kaggle dataset path

    Returns:
        pd.DataFrame: Loaded dataset
    """
    # Download dataset (returns path)
    path = kagglehub.dataset_download("mdsultanulislamovi/mental-disorders-dataset")

    # Load CSV
    df = pd.read_csv(path + "/mental_disorders_dataset.csv")
    
    print(f"Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    return df
