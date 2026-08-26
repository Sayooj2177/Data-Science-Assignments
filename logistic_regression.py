# ============================================================
# LOGISTIC REGRESSION - DIABETES PREDICTION
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    classification_report,
    confusion_matrix
)


# ============================================================
# 1. DATA EXPLORATION
# ============================================================

# Load dataset
df = pd.read_csv("diabetes.csv")

print("=" * 60)
print("DATASET INFORMATION")
print("=" * 60)

print("\nFirst 5 rows:")
print(df.head())

print("\nDataset shape:")
print(df.shape)

print("\nColumn names:")
print(df.columns.tolist())

print("\nData types:")
print(df.dtypes)

print("\nSummary statistics:")
print(df.describe())

print("\nMissing values:")
print(df.isnull().sum())


# ============================================================
# 2. HANDLE ZERO VALUES REPRESENTING MISSING VALUES
# ============================================================

# In the diabetes dataset, zero is not medically meaningful
# for these columns and can represent missing measurements.

columns_with_missing = [
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI"
]

df[columns_with_missing] = df[columns_with_missing].replace(
    0, np.nan
)

print("\nMissing values after replacing invalid zeros:")
print(df.isnull().sum())


# ============================================================
# 3. DATA VISUALIZATION
# ============================================================

# Histograms
df.hist(figsize=(12, 10), bins=20)
plt.suptitle("Distribution of Diabetes Dataset Features")
plt.tight_layout()
plt.show()


# Box plots
plt.figure(figsize=(12, 6))
sns.boxplot(data=df)
plt.xticks(rotation=45)
plt.title("Box Plot of Diabetes Dataset Features")
plt.tight_layout()
plt.show()


# Correlation heatmap
plt.figure(figsize=(10, 7))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.show()


# ============================================================
# 4. PREPARE FEATURES AND TARGET
# ============================================================

X = df.drop("Outcome", axis=1)
y = df["Outcome"]

print("\nFeature columns:")
print(X.columns.tolist())

print("\nTarget distribution:")
print(y.value_counts())


# ============================================================
# 5. TRAIN-TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining data shape:", X_train.shape)
print("Testing data shape:", X_test.shape)


# ============================================================
# 6. BUILD LOGISTIC REGRESSION PIPELINE
# ============================================================

model = Pipeline([
    (
        "imputer",
        SimpleImputer(strategy="median")
    ),
    (
        "scaler",
        StandardScaler()
    ),
    (
        "logistic_regression",
        LogisticRegression(max_iter=1000)
    )
])


# ============================================================
# 7. TRAIN MODEL
# ============================================================

model.fit(X_train, y_train)

print("\nModel training completed successfully.")


# ============================================================
# 8. MODEL PREDICTION
# ============================================================

y_pred = model.predict(X_test)
y_probability = model.predict_proba(X_test)[:, 1]


# ============================================================
# 9. MODEL EVALUATION
# ============================================================

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_probability)

print("\n" + "=" * 60)
print("MODEL EVALUATION")
print("=" * 60)

print(f"\nAccuracy  : {accuracy:.4f}")
print(f"Precision : {precision:.4f}")
print(f"Recall    : {recall:.4f}")
print(f"F1-Score  : {f1:.4f}")
print(f"ROC-AUC   : {roc_auc:.4f}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred))


# ============================================================
# 10. CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6, 5))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues"
)
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.show()


# ============================================================
# 11. ROC CURVE
# ============================================================

fpr, tpr, thresholds = roc_curve(
    y_test,
    y_probability
)

plt.figure(figsize=(7, 6))
plt.plot(
    fpr,
    tpr,
    label=f"Logistic Regression (AUC = {roc_auc:.2f})"
)
plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--"
)
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.tight_layout()
plt.show()


# ============================================================
# 12. INTERPRETATION OF MODEL COEFFICIENTS
# ============================================================

logistic_model = model.named_steps["logistic_regression"]

coefficients = pd.DataFrame({
    "Feature": X.columns,
    "Coefficient": logistic_model.coef_[0]
})

coefficients["Absolute_Coefficient"] = (
    coefficients["Coefficient"].abs()
)

coefficients = coefficients.sort_values(
    by="Absolute_Coefficient",
    ascending=False
)

print("\n" + "=" * 60)
print("LOGISTIC REGRESSION COEFFICIENTS")
print("=" * 60)

print(coefficients[["Feature", "Coefficient"]].to_string(index=False))


# ============================================================
# 13. SAVE TRAINED MODEL
# ============================================================

joblib.dump(
    model,
    "logistic_model.pkl"
)

print("\n" + "=" * 60)
print("MODEL SAVED")
print("=" * 60)

print("\nSaved file: logistic_model.pkl")
print("\nReady for Streamlit deployment!")