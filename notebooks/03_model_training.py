# Model Training & Evaluation

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Load preprocessed data
df = pd.read_csv('data/preprocessed_mental_disorders.csv')
X = df.iloc[:, 1:-1]
y = df['Expert Diagnose']

# Split into train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ------------------ Random Forest ------------------
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)

print("Random Forest Accuracy:", accuracy_score(y_test, y_pred_rf))

# Confusion Matrix
cm_rf = confusion_matrix(y_test, y_pred_rf)
plt.figure(figsize=(6,5))
sns.heatmap(cm_rf, annot=True, fmt='d', cmap='Blues')
plt.title("Random Forest Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# Feature Importance
importances = rf.feature_importances_
indices = np.argsort(importances)
features = X.columns

plt.figure(figsize=(10,6))
plt.barh(range(len(indices)), importances[indices], align='center')
plt.yticks(range(len(indices)), [features[i] for i in indices])
plt.xlabel('Importance')
plt.title('Feature Importance - Random Forest')
plt.show()

# Select important features (>0.05)
important_features = [features[i] for i in range(len(features)) if importances[i] > 0.05]
print("Selected Important Features:", important_features)

# Train RF on important features
X_train_imp = X_train[important_features]
X_test_imp = X_test[important_features]

rf_imp = RandomForestClassifier(random_state=42)
rf_imp.fit(X_train_imp, y_train)
y_pred_rf_imp = rf_imp.predict(X_test_imp)

print("Random Forest (Important Features) Accuracy:", accuracy_score(y_test, y_pred_rf_imp))

# ------------------ Multinomial Logistic Regression ------------------
lrm = LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=500)
lrm.fit(X_train_imp, y_train)
y_pred_lr_imp = lrm.predict(X_test_imp)

print("Multinomial Logistic Regression Accuracy:", accuracy_score(y_test, y_pred_lr_imp))

# ------------------ Classification Reports ------------------
print("\nRandom Forest (Important Features):")
print(classification_report(y_test, y_pred_rf_imp))

print("\nMultinomial Logistic Regression (Important Features):")
print(classification_report(y_test, y_pred_lr_imp))

# Optional: Combined Confusion Matrix Plot
fig, axes = plt.subplots(1, 2, figsize=(12,5))

sns.heatmap(confusion_matrix(y_test, y_pred_rf_imp), annot=True, fmt='d', cmap='Blues', ax=axes[0])
axes[0].set_title("Random Forest (Imp Features)")
axes[0].set_xlabel("Predicted")
axes[0].set_ylabel("Actual")

sns.heatmap(confusion_matrix(y_test, y_pred_lr_imp), annot=True, fmt='d', cmap='Blues', ax=axes[1])
axes[1].set_title("Multinomial Logistic Regression")
axes[1].set_xlabel("Predicted")
axes[1].set_ylabel("Actual")

plt.tight_layout()
plt.show()
