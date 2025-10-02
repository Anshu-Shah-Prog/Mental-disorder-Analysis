# src/feature_selection.py
"""
Feature Selection for Mental Disorder Prediction
- Extract important features using Random Forest
"""

from sklearn.ensemble import RandomForestClassifier
import pandas as pd

def select_important_features(X, y, threshold=0.05, random_state=42):
    """
    Fit Random Forest and select features above importance threshold.
    
    Args:
        X (pd.DataFrame): Feature dataframe
        y (pd.Series): Target variable
        threshold (float): Minimum importance to keep feature
        random_state (int)
    
    Returns:
        List[str]: Important feature names
    """
    rf = RandomForestClassifier(random_state=random_state)
    rf.fit(X, y)
    
    importances = rf.feature_importances_
    features = X.columns
    important_features = [features[i] for i in range(len(features)) if importances[i] > threshold]
    
    return important_features, rf
