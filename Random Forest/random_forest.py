# ============================================================
# RANDOM FOREST ASSIGNMENT - GLASS DATASET
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import (
    RandomForestClassifier,
    BaggingClassifier,
    AdaBoostClassifier
)
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

# ============================================================
# 1. DATA OVERVIEW
# ============================================================

print("=" * 60)
print("1. DATA OVERVIEW")
print("=" * 60)

# Load dataset
df = pd.read_excel("glass.xlsx", sheet_name="glass")
print(df.columns)

print("\nFirst 5 rows:")
print(df.head())

print("\nShape of dataset:")
print(df.shape)

print("\nColumn names:")
print(df.columns.tolist())

print("\nData types:")
print(df.dtypes)

print("\nStatistical summary:")
print(df.describe())

# ============================================================
# 2. EXPLORATORY DATA ANALYSIS (EDA)
# ============================================================

print("\n" + "=" * 60)
print("2. EXPLORATORY DATA ANALYSIS")
print("=" * 60)

# Missing values
print("\nMissing values:")
print(df.isnull().sum())

print("\nTotal missing values:")
print(df.isnull().sum().sum())

# Duplicate values
print("\nNumber of duplicate rows:")
print(df.duplicated().sum())

# Target class distribution
print("\nTarget class distribution:")
print(df["Type"].value_counts().sort_index())

# Check unique values
print("\nUnique values in each column:")
for column in df.columns:
    print(column, ":", df[column].nunique())

# ============================================================
# 3. OUTLIER DETECTION
# ============================================================

print("\n" + "=" * 60)
print("3. OUTLIER DETECTION")
print("=" * 60)

features = df.drop(columns=["Type"])

Q1 = features.quantile(0.25)
Q3 = features.quantile(0.75)
IQR = Q3 - Q1

lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

outliers = ((features < lower_bound) |
            (features > upper_bound)).sum()

print("\nNumber of potential outliers:")
print(outliers)

# ============================================================
# 4. DATA VISUALIZATION
# ============================================================

print("\n" + "=" * 60)
print("4. DATA VISUALIZATION")
print("=" * 60)

# Histograms
features.hist(figsize=(14, 10), bins=20)
plt.suptitle("Feature Distributions")
plt.tight_layout()
plt.show()

# Box plots
plt.figure(figsize=(14, 7))
sns.boxplot(data=features)
plt.title("Box Plot of Glass Features")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Correlation heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.show()

# Target class distribution
plt.figure(figsize=(8, 5))
sns.countplot(x="Type", data=df)
plt.title("Glass Type Distribution")
plt.xlabel("Glass Type")
plt.ylabel("Count")
plt.tight_layout()
plt.show()

# Pairplot
sns.pairplot(
    df,
    hue="Type",
    vars=["RI", "Na", "Mg", "Al", "Si"]
)
plt.show()

# ============================================================
# 5. DATA PREPROCESSING
# ============================================================

print("\n" + "=" * 60)
print("5. DATA PREPROCESSING")
print("=" * 60)

# Separate features and target
X = df.drop(columns=["Type"])
y = df["Type"]

# Check categorical variables
categorical_columns = X.select_dtypes(
    include=["object", "category"]
).columns.tolist()

print("\nCategorical columns:")
print(categorical_columns)

if len(categorical_columns) == 0:
    print("No categorical variables are present.")

# Missing value handling
imputer = SimpleImputer(strategy="median")

X_imputed = imputer.fit_transform(X)

print("\nMissing values handled using median imputation.")

# Feature scaling
scaler = StandardScaler()

X_scaled = scaler.fit_transform(X_imputed)

print("Feature scaling completed using StandardScaler.")

# ============================================================
# 6. TRAIN-TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining data shape:")
print(X_train.shape)

print("\nTesting data shape:")
print(X_test.shape)

# ============================================================
# 7. RANDOM FOREST CLASSIFIER
# ============================================================

print("\n" + "=" * 60)
print("6. RANDOM FOREST MODEL")
print("=" * 60)

rf_model = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    class_weight="balanced"
)

rf_model.fit(X_train, y_train)

# Prediction
y_pred_rf = rf_model.predict(X_test)

# Evaluation
rf_accuracy = accuracy_score(y_test, y_pred_rf)
rf_precision = precision_score(
    y_test,
    y_pred_rf,
    average="weighted",
    zero_division=0
)
rf_recall = recall_score(
    y_test,
    y_pred_rf,
    average="weighted",
    zero_division=0
)
rf_f1 = f1_score(
    y_test,
    y_pred_rf,
    average="weighted",
    zero_division=0
)

print("\nRandom Forest Results")
print("-" * 40)
print(f"Accuracy  : {rf_accuracy:.4f}")
print(f"Precision : {rf_precision:.4f}")
print(f"Recall    : {rf_recall:.4f}")
print(f"F1 Score  : {rf_f1:.4f}")

print("\nClassification Report:")
print(classification_report(
    y_test,
    y_pred_rf,
    zero_division=0
))

# Confusion matrix
cm_rf = confusion_matrix(y_test, y_pred_rf)

plt.figure(figsize=(8, 6))
sns.heatmap(
    cm_rf,
    annot=True,
    fmt="d",
    cmap="Blues"
)
plt.title("Random Forest Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.show()

# ============================================================
# 8. FEATURE IMPORTANCE
# ============================================================

print("\n" + "=" * 60)
print("7. FEATURE IMPORTANCE")
print("=" * 60)

importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": rf_model.feature_importances_
})

importance = importance.sort_values(
    by="Importance",
    ascending=False
)

print("\nFeature Importance:")
print(importance)

plt.figure(figsize=(10, 6))
sns.barplot(
    x="Importance",
    y="Feature",
    data=importance
)
plt.title("Random Forest Feature Importance")
plt.tight_layout()
plt.show()

# ============================================================
# 9. BAGGING CLASSIFIER
# ============================================================

print("\n" + "=" * 60)
print("8. BAGGING METHOD")
print("=" * 60)

bagging_model = BaggingClassifier(
    n_estimators=200,
    random_state=42
)

bagging_model.fit(X_train, y_train)

y_pred_bagging = bagging_model.predict(X_test)

bagging_accuracy = accuracy_score(
    y_test,
    y_pred_bagging
)

bagging_precision = precision_score(
    y_test,
    y_pred_bagging,
    average="weighted",
    zero_division=0
)

bagging_recall = recall_score(
    y_test,
    y_pred_bagging,
    average="weighted",
    zero_division=0
)

bagging_f1 = f1_score(
    y_test,
    y_pred_bagging,
    average="weighted",
    zero_division=0
)

print("\nBagging Results")
print("-" * 40)
print(f"Accuracy  : {bagging_accuracy:.4f}")
print(f"Precision : {bagging_precision:.4f}")
print(f"Recall    : {bagging_recall:.4f}")
print(f"F1 Score  : {bagging_f1:.4f}")

# ============================================================
# 10. BOOSTING - ADABOOST
# ============================================================

print("\n" + "=" * 60)
print("9. BOOSTING METHOD")
print("=" * 60)

boosting_model = AdaBoostClassifier(
    n_estimators=100,
    random_state=42
)

boosting_model.fit(X_train, y_train)

y_pred_boosting = boosting_model.predict(X_test)

boosting_accuracy = accuracy_score(
    y_test,
    y_pred_boosting
)

boosting_precision = precision_score(
    y_test,
    y_pred_boosting,
    average="weighted",
    zero_division=0
)

boosting_recall = recall_score(
    y_test,
    y_pred_boosting,
    average="weighted",
    zero_division=0
)

boosting_f1 = f1_score(
    y_test,
    y_pred_boosting,
    average="weighted",
    zero_division=0
)

print("\nAdaBoost Results")
print("-" * 40)
print(f"Accuracy  : {boosting_accuracy:.4f}")
print(f"Precision : {boosting_precision:.4f}")
print(f"Recall    : {boosting_recall:.4f}")
print(f"F1 Score  : {boosting_f1:.4f}")

# ============================================================
# 11. BAGGING VS BOOSTING VS RANDOM FOREST
# ============================================================

print("\n" + "=" * 60)
print("10. MODEL COMPARISON")
print("=" * 60)

comparison = pd.DataFrame({
    "Model": [
        "Random Forest",
        "Bagging",
        "AdaBoost"
    ],
    "Accuracy": [
        rf_accuracy,
        bagging_accuracy,
        boosting_accuracy
    ],
    "Precision": [
        rf_precision,
        bagging_precision,
        boosting_precision
    ],
    "Recall": [
        rf_recall,
        bagging_recall,
        boosting_recall
    ],
    "F1 Score": [
        rf_f1,
        bagging_f1,
        boosting_f1
    ]
})

print("\nModel Comparison:")
print(comparison.to_string(index=False))

# Comparison graph
comparison.set_index("Model").plot(
    kind="bar",
    figsize=(12, 7)
)

plt.title("Random Forest vs Bagging vs AdaBoost")
plt.ylabel("Score")
plt.ylim(0, 1)
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()

# ============================================================
# 12. IMBALANCE HANDLING
# ============================================================

print("\n" + "=" * 60)
print("11. HANDLING IMBALANCED DATA")
print("=" * 60)

class_counts = y.value_counts().sort_index()

print("\nClass distribution:")
print(class_counts)

print(
    "\nRandom Forest was implemented with "
    "class_weight='balanced' to give greater importance "
    "to minority classes."
)

print(
    "\nStratified train-test split was also used to preserve "
    "the class distribution in training and testing data."
)

# ============================================================
# 13. INTERVIEW QUESTIONS
# ============================================================

print("\n" + "=" * 60)
print("12. INTERVIEW QUESTIONS")
print("=" * 60)

print("""
Q1. What is the difference between Bagging and Boosting?

Answer:
Bagging stands for Bootstrap Aggregating. It creates multiple
models independently using different bootstrap samples of the
training data and combines their predictions. Random Forest is
an example of a bagging-based ensemble method.

Boosting creates models sequentially. Each new model focuses
more on the errors made by the previous models. AdaBoost and
Gradient Boosting are examples of boosting methods.

The main difference is that Bagging trains models independently,
while Boosting trains models sequentially. Bagging mainly reduces
variance, whereas Boosting mainly focuses on reducing bias and
improving predictive performance.


Q2. How can you handle imbalanced data in a classification problem?

Answer:
Imbalanced data occurs when some classes have significantly more
observations than other classes.

Common techniques for handling imbalance include:

1. Oversampling the minority class.
2. Undersampling the majority class.
3. Using SMOTE to generate synthetic minority samples.
4. Using class weights in machine learning models.
5. Using evaluation metrics such as precision, recall and F1-score
   instead of relying only on accuracy.

In this assignment, class_weight='balanced' was used in the
Random Forest model. Stratified train-test splitting was also
used so that the class distribution was maintained in both
training and testing datasets.
""")

print("\n" + "=" * 60)
print("RANDOM FOREST ASSIGNMENT COMPLETED")
print("=" * 60)