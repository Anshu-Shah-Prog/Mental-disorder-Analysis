import numpy as np
import pandas as pd
import kagglehub
from matplotlib import pyplot as plt
import seaborn as sns

# Loading dataset from kaggle to colab
path = kagglehub.dataset_download("mdsultanulislamovi/mental-disorders-dataset")
df = pd.read_csv(path + "/mental_disorders_dataset.csv")

#first  five rows of dataset
df.head()

df.info()
# #Total Records: 120 patients
# Total Features: 19 columns
# Symptom Features: 17 behavioral/psychological indicators
# Target Variable: Expert Diagnose

df.describe()


# donut plot of Target variable (Expert Diagnose ) 
plt.figure(figsize=(4,4))
custom_colors = ['#0077b6', '#0096c7', '#00b4d8', '#48cae4']
plt.pie(df['Expert Diagnose'].value_counts(), labels=df['Expert Diagnose'].value_counts().index, colors=custom_colors, autopct='%1.1f%%', startangle=90)
# Add a circle in the center for the donut hole
centre_circle = plt.Circle((0,0),0.6,fc='white')
plt.gca().add_artist(centre_circle)
plt.title('Expert Diagnose')
plt.axis('equal')  # make the pie circular
plt.show()
