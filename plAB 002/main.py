# Step 1: Data Exploration and Preprocessing

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE  # For oversampling imbalanced classes
from imblearn.under_sampling import RandomUnderSampler  # For undersampling imbalanced classes

# Load your dataset (replace 'camtel_data.csv' with your actual file)
data = pd.read_csv('camtel_data.csv')

# Display the first few rows of the dataset
print(data.head())

# Check for missing values
print("\nMissing Values:\n", data.isnull().sum())

# Visualize missing data (if needed)
import missingno as msno
msno.matrix(data)
plt.show()

# Handle missing values (if any) - Example: Impute with median for numerical, mode for categorical
data['age'] = data['age'].fillna(data['age'].median())  # Impute numerical
data['region'] = data['region'].fillna(data['region'].mode()[0])  # Impute categorical

# Check for correlations between features
correlation_matrix = data.corr()
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()

# Split the dataset into training and testing based on different time periods
# Assuming 'date' column exists to split the data by time
data['date'] = pd.to_datetime(data['date'])  # Convert date to datetime type
train_data = data[data['date'] < '2022-01-01']  # First two years for training
test_data = data[data['date'] >= '2022-01-01']  # Last year for testing

# Define features and target
X_train = train_data.drop(['churn', 'date'], axis=1)
y_train = train_data['churn']
X_test = test_data.drop(['churn', 'date'], axis=1)
y_test = test_data['churn']

# Check for class imbalance
print(f"Class distribution in train set:\n{y_train.value_counts()}")
print(f"Class distribution in test set:\n{y_test.value_counts()}")
