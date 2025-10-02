# src/model_training.py
"""
Model Training & Evaluation
- Random Forest
- Multinomial Logistic Regression
- Confusion Matrix & Classification Report
"""

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

def train_random_forest(X_train, y_train, X_test, y_test, random_state=42):
    """
    Train Random Forest and evaluate model.
    
    Returns:
        model, y_pred, metrics_dict
    """
    rf = RandomForestClassifier(random_state=random_state)
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)
    
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "classification_report": classification_report(y_test, y_pred),
        "confusion_matrix": confusion_matrix(y_test, y_pred)
    }
    return rf, y_pred, metrics

def train_logistic_regression(X_train, y_train, X_test, y_test):
    """
    Train Multinomial Logistic Regression and evaluate model.
    
    Returns:
        model, y_pred, metrics_dict
    """
    lrm = LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=500)
    lrm.fit(X_train, y_train)
    y_pred = lrm.predict(X_test)
    
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "classification_report": classification_report(y_test, y_pred),
        "confusion_matrix": confusion
