# 01_data_exploration.ipynb
# Exploratory Data Analysis (EDA) using src modules

# Import libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from src.kaggle_data import load_mental_disorders_dataset
from src.data_preprocessing import preprocess_data


# Load dataset from Kaggle
df_raw = load_mental_disorders_dataset()



# Display dataset info
df_raw.head()
df_raw.info()
df_raw.describe()

# Preprocess for EDA
df = preprocess_data(df_raw)

# Display processed dataset info
df.head()
df.info()
df.describe()

# Donut plot for Expert Diagnose distribution
plt.figure(figsize=(5,5))
colors = ['#0077b6', '#0096c7', '#00b4d8', '#48cae4']
plt.pie(df['Expert Diagnose'].value_counts(), 
        labels=df['Expert Diagnose'].value_counts().index,
        colors=colors, autopct='%1.1f%%', startangle=90)
centre_circle = plt.Circle((0,0),0.6,fc='white')
plt.gca().add_artist(centre_circle)
plt.title('Expert Diagnose Distribution')
plt.axis('equal')
plt.show()

# Example count plot for a key feature
sns.countplot(data=df, x='Sleep dissorder', hue='Expert Diagnose', palette='Blues')
plt.title('Sleep Disorder vs Diagnosis')
plt.show()
