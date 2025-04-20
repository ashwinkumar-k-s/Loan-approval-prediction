# Loan Approval Prediction - Complete Clean Code

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, mean_absolute_error, mean_squared_error, r2_score, confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc
import joblib

#  Load Dataset
data = pd.read_csv(r"D:\loan_approval_pred\loan_approval_dataset.csv")
print(" Data Loaded Successfully!\n")
print(data.info())

#  Clean Column Names
data.columns = data.columns.str.strip()

# Missing Value Analysis
print("\nMissing Values in Dataset:\n")
print(data.isnull().sum())

# Handle Missing Values
# Impute numeric columns with median
num_cols = data.select_dtypes(include=['int64', 'float64']).columns
for col in num_cols:
    if data[col].isnull().sum() > 0:
        data[col].fillna(data[col].median(), inplace=True)

# Impute categorical columns with mode
cat_cols = data.select_dtypes(include='object').columns
for col in cat_cols:
    if data[col].isnull().sum() > 0:
        data[col].fillna(data[col].mode()[0], inplace=True)

print("\nMissing Values After Analysis:\n")
print(data.isnull().sum())


#  Scale Numeric Data
# Unified scaling for all numerical columns
scaler = MinMaxScaler()
num_cols = data.select_dtypes(include='int64').columns
data[num_cols] = scaler.fit_transform(data[num_cols])

# Save the scaler for Streamlit
joblib.dump(scaler, r"D:\loan_approval_pred\ml_model\scaler.pkl")


#  Encode Categorical Data
label_encoder = LabelEncoder()
for col in data.columns[data.dtypes == 'object']:
    data[col] = label_encoder.fit_transform(data[col])

print("\n Data Preprocessing Completed!")
print(data.head())

#  Exploratory Data Analysis (EDA)

# Loan Status Count Plot
plt.figure(figsize=(6, 4))
sns.countplot(x='loan_status', data=data)
plt.xlabel("Loan Status (0 = Rejected, 1 = Approved)")
plt.ylabel("Count")
plt.title("Loan Approval Status Distribution")
plt.show()

# Correlation Matrix Heatmap
plt.figure(figsize=(15, 15))
sns.heatmap(data.corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.show()

# Sorted Correlation Values
print("\n Feature Correlation with Loan Status:\n")
print(data.corr()['loan_status'].sort_values(ascending=False))

#  Feature Importance - Random Forest
X = data.drop(columns='loan_status')
y = data['loan_status']

rf_importance_model = RandomForestClassifier(random_state=42)
rf_importance_model.fit(X, y)

importance = rf_importance_model.feature_importances_
feature_names = X.columns

plt.figure(figsize=(8, 5))
sns.barplot(x=importance, y=feature_names, palette='viridis')
plt.title('Feature Importance (Random Forest)')
plt.show()

# Dropping Low-Correlation Features
drop_columns = [
    'commercial_assets_value', 'self_employed', 'loan_id',
    'education', 'bank_asset_value', 'residential_assets_value'
]
data.drop(columns=drop_columns, inplace=True, errors='ignore')

#  Feature Importance - Random Forest
X = data.drop(columns='loan_status')
y = data['loan_status']

rf_importance_model = RandomForestClassifier(random_state=42)
rf_importance_model.fit(X, y)

importance = rf_importance_model.feature_importances_
feature_names = X.columns

plt.figure(figsize=(8, 5))
sns.barplot(x=importance, y=feature_names, palette='viridis')
plt.title('Feature Importance (Random Forest)')
plt.show()

# Pairplot for Key Features
key_features = ['loan_status', 'cibil_score', 'loan_term', 'income_annum',
                'no_of_dependents', 'loan_amount', 'luxury_assets_value']
sns.pairplot(data[key_features], hue='loan_status', palette='viridis')
plt.show()

#  Model Training & Evaluation

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Logistic Regression
lr_model = LogisticRegression(max_iter=1000)
lr_model.fit(X_train, y_train)
lr_pred = lr_model.predict(X_test)

# Decision Tree Classifier
dt_model = DecisionTreeClassifier(random_state=42)
dt_model.fit(X_train, y_train)
dt_pred = dt_model.predict(X_test)

# Random Forest Classifier
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
rf_pred = rf_model.predict(X_test)

# Function to Print Model Metrics
def evaluate(name, y_true, y_pred):
    print(f"\n {name} Evaluation:")
    print(classification_report(y_true, y_pred))
    print(f"Accuracy: {accuracy_score(y_true, y_pred) * 100:.2f}%")
    print(f"Mean Absolute Error: {mean_absolute_error(y_true, y_pred):.4f}")
    print(f"Mean Squared Error: {mean_squared_error(y_true, y_pred):.4f}")
    print(f"Root Mean Squared Error: {np.sqrt(mean_squared_error(y_true, y_pred)):.4f}")
    print(f"R2 Score: {r2_score(y_true, y_pred):.4f}")

# Evaluate All Models
evaluate("Logistic Regression", y_test, lr_pred)
evaluate("Decision Tree", y_test, dt_pred)
evaluate("Random Forest", y_test, rf_pred)

# Accuracy Comparison Plot
models = ['Logistic Regression', 'Decision Tree', 'Random Forest']
scores = [
    accuracy_score(y_test, lr_pred) * 100,
    accuracy_score(y_test, dt_pred) * 100,
    accuracy_score(y_test, rf_pred) * 100
]

plt.figure(figsize=(8, 5))
sns.barplot(x=models, y=scores, palette='viridis')
plt.ylabel("Accuracy (%)")
plt.title("Model Accuracy Comparison")
plt.ylim(0, 100)
plt.show()

# Logistic Regression Confusion Matrix
cm_lr = confusion_matrix(y_test, lr_pred)
disp_lr = ConfusionMatrixDisplay(confusion_matrix=cm_lr, display_labels=['Rejected', 'Approved'])
disp_lr.plot(cmap='Blues')
plt.title("Logistic Regression - Confusion Matrix")
plt.show()

# Decision Tree Confusion Matrix
cm_dt = confusion_matrix(y_test, dt_pred)
disp_dt = ConfusionMatrixDisplay(confusion_matrix=cm_dt, display_labels=['Rejected', 'Approved'])
disp_dt.plot(cmap='Greens')
plt.title("Decision Tree - Confusion Matrix")
plt.show()

# Random Forest Confusion Matrix
cm_rf = confusion_matrix(y_test, rf_pred)
disp_rf = ConfusionMatrixDisplay(confusion_matrix=cm_rf, display_labels=['Rejected', 'Approved'])
disp_rf.plot(cmap='Oranges')
plt.title("Random Forest - Confusion Matrix")
plt.show()

# Predict probabilities (instead of class labels!)
lr_probs = lr_model.predict_proba(X_test)[:, 1]
dt_probs = dt_model.predict_proba(X_test)[:, 1]
rf_probs = rf_model.predict_proba(X_test)[:, 1]

# Calculate ROC Curve and AUC for each
fpr_lr, tpr_lr, _ = roc_curve(y_test, lr_probs)
fpr_dt, tpr_dt, _ = roc_curve(y_test, dt_probs)
fpr_rf, tpr_rf, _ = roc_curve(y_test, rf_probs)

auc_lr = auc(fpr_lr, tpr_lr)
auc_dt = auc(fpr_dt, tpr_dt)
auc_rf = auc(fpr_rf, tpr_rf)

# Plot ROC Curves
plt.figure(figsize=(8, 6))
plt.plot(fpr_lr, tpr_lr, label=f"Logistic Regression (AUC = {auc_lr:.2f})")
plt.plot(fpr_dt, tpr_dt, label=f"Decision Tree (AUC = {auc_dt:.2f})")
plt.plot(fpr_rf, tpr_rf, label=f"Random Forest (AUC = {auc_rf:.2f})")

# Diagonal line for reference
plt.plot([0, 1], [0, 1], 'k--', label="Random Classifier")

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve Comparison")
plt.legend(loc="lower right")
plt.grid(True)
plt.show()


# Save Models for Deployment
joblib.dump(lr_model, r"D:\loan_approval_pred\ml_model\lr_model.pkl")
joblib.dump(dt_model, r"D:\loan_approval_pred\ml_model\dtree_model.pkl")
joblib.dump(rf_model, r"D:\loan_approval_pred\ml_model\rf_model.pkl")

print("\n Models saved successfully!")


