# 03_model_training.ipynb
# Model Training & Evaluation using src modules

import pandas as pd
from sklearn.model_selection import train_test_split
from src.feature_selection import select_important_features
from src.model_training import train_random_forest, train_logistic_regression
import matplotlib.pyplot as plt
import seaborn as sns

# Load preprocessed data
X = pd.read_csv('data/X_preprocessed.csv')
y = pd.read_csv('data/y_preprocessed.csv').squeeze()  # convert to Series

# Split train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ------------------ Feature Selection ------------------
important_features, rf_model_full = select_important_features(X_train, y_train)
print("Selected Important Features:", important_features)

X_train_imp = X_train[important_features]
X_test_imp = X_test[important_features]

# ------------------ Random Forest ------------------
rf_imp, y_pred_rf_imp, rf_metrics = train_random_forest(X_train_imp, y_train, X_test_imp, y_test)
print("Random Forest Accuracy:", rf_metrics['accuracy'])
print(rf_metrics['classification_report'])

# Confusion Matrix Plot
plt.figure(figsize=(6,5))
sns.heatmap(rf_metrics['confusion_matrix'], annot=True, fmt='d', cmap='Blues')
plt.title("Random Forest (Important Features) Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# ------------------ Multinomial Logistic Regression ------------------
lr_model_imp, y_pred_lr_imp, lr_metrics = train_logistic_regression(X_train_imp, y_train, X_test_imp, y_test)
print("Logistic Regression Accuracy:", lr_metrics['accuracy'])
print(lr_metrics['classification_report'])

# Combined Confusion Matrix Plot
fig, axes = plt.subplots(1, 2, figsize=(12,5))
sns.heatmap(rf_metrics['confusion_matrix'], annot=True, fmt='d', cmap='Blues', ax=axes[0])
axes[0].set_title("Random Forest")
axes[0].set_xlabel("Predicted")
axes[0].set_ylabel("Actual")

sns.heatmap(lr_metrics['confusion_matrix'], annot=True, fmt='d', cmap='Blues', ax=axes[1])
axes[1].set_title("Multinomial Logistic Regression")
axes[1].set_xlabel("Predicted")
axes[1].set_ylabel("Actual")
plt.tight_layout()
plt.show()
