"""
Train and export the Cardiovascular Heart Disease Risk Prediction model.
Performs clinical cardiology feature engineering, robust feature scaling,
benchmarks multiple classifiers (Random Forest, SVM, Logistic Regression, Gradient Boosting),
evaluates clinical sensitivity/recall and ROC-AUC, and exports production model artifacts.
"""

import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
import os
import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)

# 1. Load Dataset
data_path = os.path.join(os.path.dirname(__file__), 'data', 'heart.csv')
if not os.path.exists(data_path):
    data_path = os.path.join(os.path.dirname(__file__), 'heart.csv')

df = pd.read_csv(data_path)
print(f"Loaded clinical heart dataset! Shape: {df.shape}")

# 2. Clinical Feature Engineering (Cardiology Domain Logic)
# Expected Max Heart Rate formula (Haskell & Fox): 220 - age
df['max_hr_expected'] = 220 - df['age']
# Heart Rate Reserve Ratio (fraction of expected max achieved during peak stress)
df['hr_reserve_ratio'] = df['thalach'] / df['max_hr_expected']
# Blood Pressure to Serum Cholesterol Ratio
df['bp_chol_ratio'] = df['trestbps'] / df['chol']
# ST Depression to Slope interaction (Exercise stress test marker)
df['st_slope_interaction'] = df['oldpeak'] * (df['slope'] + 1)

print("Engineered clinical features: max_hr_expected, hr_reserve_ratio, bp_chol_ratio, st_slope_interaction")

# 3. Define Features & Target
X = df.drop(columns=['target'])
y = df['target']

disease_cases = (y == 1).sum()
healthy_cases = (y == 0).sum()
print(f"Target Distribution: Heart Disease = {disease_cases} ({disease_cases/len(df)*100:.1f}%), Healthy = {healthy_cases} ({healthy_cases/len(df)*100:.1f}%)")

# 4. Stratified Train / Test Split (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=100, stratify=y
)
print(f"Train split: {X_train.shape[0]} patients | Test split: {X_test.shape[0]} patients")

# 5. Feature Scaling (StandardScaler)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 6. Benchmark Multiple Classifiers
models = {
    'Support Vector Machine (RBF)': SVC(kernel='rbf', C=1.5, probability=True, random_state=42),
    'Random Forest Classifier': RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42),
    'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, max_depth=3, learning_rate=0.08, random_state=42),
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42)
}

print("\n" + "="*75)
print(f"{'Algorithm':<28} | {'Accuracy':<8} | {'Sensitivity':<11} | {'F1-Score':<8} | {'ROC-AUC':<8}")
print("="*75)

best_model = None
best_score = 0.0

for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    y_prob = model.predict_proba(X_test_scaled)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)

    print(f"{name:<28} | {acc*100:>7.2f}% | {rec*100:>10.2f}% | {f1*100:>7.2f}% | {auc*100:>7.2f}%")

    # Score combining high accuracy and high sensitivity
    composite = acc * 0.5 + rec * 0.5
    if composite > best_score:
        best_score = composite
        best_model = model

print("="*75)

# 7. Detailed Evaluation of Selected Model
y_pred_best = best_model.predict(X_test_scaled)
y_prob_best = best_model.predict_proba(X_test_scaled)[:, 1]
cm = confusion_matrix(y_test, y_pred_best)

print(f"\nSelected Diagnostic Model: {type(best_model).__name__}")
print(f"  Accuracy:    {accuracy_score(y_test, y_pred_best)*100:.2f}%")
print(f"  Sensitivity (Recall): {recall_score(y_test, y_pred_best)*100:.2f}% (Catches heart disease patients)")
print(f"  ROC-AUC:     {roc_auc_score(y_test, y_prob_best)*100:.2f}%")
print(f"  Confusion Matrix:")
print(f"    True Healthy (Correct Negative):     {cm[0][0]}")
print(f"    False Alarm (Healthy called sick):   {cm[0][1]}")
print(f"    Missed Diagnosis (False Negative):   {cm[1][0]}")
print(f"    Correct Heart Disease (True Pos):    {cm[1][1]}")

# 8. Export Production Model and Scaler
model_dir = os.path.join(os.path.dirname(__file__), 'models')
os.makedirs(model_dir, exist_ok=True)

scaler_file = os.path.join(model_dir, 'preprocessor.pkl')
model_file = os.path.join(model_dir, 'heart_model.pkl')

joblib.dump(scaler, scaler_file)
joblib.dump(best_model, model_file)

print(f"\nSaved clinical preprocessor to: {scaler_file}")
print(f"Saved trained diagnostic model to: {model_file}")

# 9. Verification Clinical Test on a High-Risk Cardiac Patient
high_risk_patient = pd.DataFrame([{
    'age': 63,
    'sex': 1,
    'cp': 3,
    'trestbps': 145,
    'chol': 233,
    'fbs': 1,
    'restecg': 0,
    'thalach': 150,
    'exang': 0,
    'oldpeak': 2.3,
    'slope': 0,
    'ca': 0,
    'thal': 1,
    'max_hr_expected': 220 - 63,
    'hr_reserve_ratio': 150 / (220 - 63),
    'bp_chol_ratio': 145 / 233,
    'st_slope_interaction': 2.3 * (0 + 1)
}])

loaded_scaler = joblib.load(scaler_file)
loaded_model = joblib.load(model_file)

sample_scaled = loaded_scaler.transform(high_risk_patient)
pred_diagnosis = loaded_model.predict(sample_scaled)[0]
cardiac_risk_prob = loaded_model.predict_proba(sample_scaled)[0][1]

status = "POSITIVE: HIGH RISK OF CARDIAC DISEASE (Action Required)" if pred_diagnosis == 1 else "NEGATIVE: NORMAL / LOW RISK"
print(f"\nVerification Test on Cardiac Patient:")
print(f"  Diagnosis:    {status}")
print(f"  Risk Score:   {cardiac_risk_prob*100:.1f}% confidence")
