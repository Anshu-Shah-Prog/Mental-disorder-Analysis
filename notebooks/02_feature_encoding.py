# 02_feature_encoding.ipynb
# Feature Encoding & Dataset Preparation using src modules

from src.kaggle_data import load_mental_disorders_dataset
from src.data_preprocessing import preprocess_data, split_features_target

# Load and preprocess dataset
df_raw = load_mental_disorders_dataset()
df = preprocess_data(df_raw)

# Split features and target
X, y = split_features_target(df)

# Save preprocessed features and target for modeling
X.to_csv('data/X_preprocessed.csv', index=False)
y.to_csv('data/y_preprocessed.csv', index=False)

print("Preprocessing complete. Features and target saved.")
