# Feature Encoding & Preprocessing

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import kagglehub

# Load dataset
path = kagglehub.dataset_download("mdsultanulislamovi/mental-disorders-dataset")
df = pd.read_csv(path + "/mental_disorders_dataset.csv")

# Encode ordinal variables
ordinal_map = {'Seldom':1, 'Sometimes':2, 'Usually':3, 'Most-Often':4}
ordinal_features = ['Sadness', 'Euphoric', 'Exhausted', 'Sleep dissorder']
for col in ordinal_features:
    df[col] = df[col].map(ordinal_map)

# Extract numeric scale values from specific columns
for col in df.columns[-4:-1]:
    df[col] = df[col].astype(str).str.extract(r'(\d+)')

# Label encoding for categorical variables
lb = LabelEncoder()
for i in range(5, len(df.columns)-4):
    df.iloc[:, i] = lb.fit_transform(df.iloc[:, i])

# Encode target variable
df['Expert Diagnose'] = lb.fit_transform(df['Expert Diagnose'])

# Split features and target
X = df.iloc[:, 1:-1]
y = df['Expert Diagnose']

# Save preprocessed dataset for modeling
df.to_csv('data/preprocessed_mental_disorders.csv', index=False)
print("Preprocessing complete. Dataset saved to 'data/preprocessed_mental_disorders.csv'")
