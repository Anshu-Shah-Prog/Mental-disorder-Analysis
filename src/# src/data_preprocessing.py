# src/data_preprocessing.py
"""
Data Preprocessing for Mental Disorder Prediction
- Ordinal encoding
- Label encoding
- Numeric extraction
"""

import pandas as pd
from sklearn.preprocessing import LabelEncoder

def preprocess_data(file_path):
    """
    Loads and preprocesses mental disorders dataset.
    
    Args:
        file_path (str): Path to CSV dataset
    
    Returns:
        pd.DataFrame: Preprocessed dataframe
    """
    df = pd.read_csv(file_path)
    
    # Encode ordinal variables
    ordinal_map = {'Seldom':1, 'Sometimes':2, 'Usually':3, 'Most-Often':4}
    ordinal_features = ['Sadness', 'Euphoric', 'Exhausted', 'Sleep dissorder']
    for col in ordinal_features:
        df[col] = df[col].map(ordinal_map)
    
    # Extract numeric values from specific columns
    for col in df.columns[-4:-1]:
        df[col] = df[col].astype(str).str.extract(r'(\d+)')
    
    # Label encode categorical features
    lb = LabelEncoder()
    for i in range(5, len(df.columns)-4):
        df.iloc[:, i] = lb.fit_transform(df.iloc[:, i])
    
    # Encode target variable
    df['Expert Diagnose'] = lb.fit_transform(df['Expert Diagnose'])
    
    return df

def split_features_target(df):
    """
    Splits dataframe into features and target.
    
    Args:
        df (pd.DataFrame)
        
    Returns:
        X, y: Features and target
    """
    X = df.iloc[:, 1:-1]
    y = df['Expert Diagnose']
    return X, y
