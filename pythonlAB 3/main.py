# Importing Required Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import VotingClassifier
from skmultiflow.drift_detection import ADWIN

# Load Dataset (replace 'camtel_data.csv' with your dataset path)
data = pd.read_csv('camtel_data.csv')

# 1. Data Exploration and Preprocessing
print("\nMissing Values:\n", data.isnull().sum())
print("\nData Overview:\n", data.describe())

# Fill missing values (example: median for numerical, mode for categorical)
data['age'] = data['age'].fillna(data['age'].median())
data['region'] = data['region'].fillna(data['region'].mode()[0])

# Check correlation between features
correlation_matrix = data.corr()
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()

# Split data into time-based training and testing sets
data['date'] = pd.to_datetime(data['date'])
train_data = data[data['date'] < '2022-01-01']
test_data = data[data['date'] >= '2022-01-01']

X_train = train_data.drop(['churn', 'date'], axis=1)
y_train = train_data['churn']
X_test = test_data.drop(['churn', 'date'], axis=1)
y_test = test_data['churn']

# Handle class imbalance using SMOTE (oversampling)
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

# 2. Initial Model Training: Logistic Regression
model = LogisticRegression(random_state=42)
model.fit(X_train_resampled, y_train_resampled)

# Evaluate on test data
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)
roc_auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])

# Print evaluation metrics
print(f"Initial Model Accuracy: {accuracy:.4f}")
print(f"Confusion Matrix:\n{conf_matrix}")
print(f"Classification Report:\n{class_report}")
print(f"AUC-ROC: {roc_auc:.4f}")

# 3. Dealing with Concept Drift and Data Shifts
# 3.1 Time-Weighted Learning (giving more weight to recent data)
time_weights = np.linspace(1, 5, len(X_train_resampled))  # Increase weight for recent data
model.fit(X_train_resampled, y_train_resampled, sample_weight=time_weights)

# 3.2 Online Learning with Stochastic Gradient Descent (SGD)
sgd_model = SGDClassifier(loss='log', random_state=42)
sgd_model.fit(X_train_resampled, y_train_resampled)

# Evaluate SGD model
y_pred_sgd = sgd_model.predict(X_test)
accuracy_sgd = accuracy_score(y_test, y_pred_sgd)
roc_auc_sgd = roc_auc_score(y_test, sgd_model.predict_proba(X_test)[:, 1])

print(f"SGD Classifier Accuracy: {accuracy_sgd:.4f}")
print(f"SGD Classifier AUC-ROC: {roc_auc_sgd:.4f}")

# 3.3 Ensemble Learning: Combine models trained on different time periods (yearly)
# For illustration, training separate models for 2020 and 2021
train_data_2020 = data[data['date'].dt.year == 2020]
train_data_2021 = data[data['date'].dt.year == 2021]

X_train_2020 = train_data_2020.drop(['churn', 'date'], axis=1)
y_train_2020 = train_data_2020['churn']

X_train_2021 = train_data_2021.drop(['churn', 'date'], axis=1)
y_train_2021 = train_data_2021['churn']

# Train models for each year
model_2020 = LogisticRegression(random_state=42)
model_2020.fit(X_train_2020, y_train_2020)

model_2021 = LogisticRegression(random_state=42)
model_2021.fit(X_train_2021, y_train_2021)

# Create an ensemble model
ensemble_model = VotingClassifier(estimators=[('2020', model_2020), ('2021', model_2021)], voting='soft')
ensemble_model.fit(X_train, y_train)

# Evaluate ensemble model
y_pred_ensemble = ensemble_model.predict(X_test)
accuracy_ensemble = accuracy_score(y_test, y_pred_ensemble)
roc_auc_ensemble = roc_auc_score(y_test, ensemble_model.predict_proba(X_test)[:, 1])

print(f"Ensemble Model Accuracy: {accuracy_ensemble:.4f}")
print(f"Ensemble Model AUC-ROC: {roc_auc_ensemble:.4f}")

# 4. Evaluate Model Adaptation (Assess performance over different time periods)
years = [2020, 2021, 2022]
for year in years:
    test_data_year = data[data['date'].dt.year == year]
    X_test_year = test_data_year.drop(['churn', 'date'], axis=1)
    y_test_year = test_data_year['churn']

    y_pred_year = model.predict(X_test_year)
    accuracy_year = accuracy_score(y_test_year, y_pred_year)
    print(f"Year {year} Model Accuracy: {accuracy_year:.4f}")

# 5. Drift Detection (Optional: Implement ADWIN for concept drift detection)
adwin = ADWIN()

# Process data point by point (streaming simulation)
for x, y in zip(X_test.values, y_test.values):
    adwin.add_element(model.predict([x]))  # Add prediction for current data point

    if adwin.detected_change():
        print("Concept drift detected at this point.")

