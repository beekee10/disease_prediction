# 🫀 Cardiovascular Health Risk Prediction (Clinical Machine Learning)

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/Library-Scikit--Learn-orange.svg)](https://scikit-learn.org/)
[![Model](https://img.shields.io/badge/Model-Support%20Vector%20Machine-brightgreen.svg)](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html)
[![Accuracy](https://img.shields.io/badge/Accuracy-90.16%25-success.svg)]()
[![Sensitivity](https://img.shields.io/badge/Sensitivity%20(Recall)-93.94%25-red.svg)]()
[![ROC-AUC](https://img.shields.io/badge/ROC--AUC-91.56%25-blueviolet.svg)]()

An end-to-end clinical Machine Learning classification project designed to identify patients at high risk of cardiovascular disease using routine, non-invasive clinical attributes.

---

## 1. Clinical Problem & Motivation

Cardiovascular diseases (CVDs) are the leading cause of death worldwide, responsible for an estimated 17.9 million lives each year. Early, accurate non-invasive diagnosis allows healthcare providers to implement preventative lifestyle and pharmacological interventions before severe cardiac events occur.

This project implements a leak-free clinical ML pipeline that achieves **90.16% diagnostic accuracy**, **93.94% clinical sensitivity (recall)**, and an **ROC-AUC of 91.56%** on held-out patient test data.

---

## 2. Dataset Overview

* **Source**: Cleveland Heart Disease Database (UCI Machine Learning Repository)
* **Patient Records**: 303 patient evaluations
* **Target Variable**: `target` (`1` = Heart Disease Present, `0` = Healthy)
* **Class Distribution**: 54.5% Positive (`1`), 45.5% Negative (`0`)

### Clinical Attribute Definitions
| Attribute | Description | Clinical Relevance |
|---|---|---|
| `age` | Patient age in years | Non-modifiable cardiovascular risk factor |
| `sex` | Biological sex (1 = male, 0 = female) | Hormonal and epidemiological risk baseline |
| `cp` | Chest pain type (0–3) | Typical vs. atypical angina assessment |
| `trestbps` | Resting blood pressure (mm Hg) | Systemic vascular resistance indicator |
| `chol` | Serum cholesterol (mg/dl) | Atherosclerotic plaque biomarker |
| `fbs` | Fasting blood sugar > 120 mg/dl | Diabetes comorbidity screening |
| `restecg` | Resting ECG results (0–2) | Left ventricular hypertrophy / repolarization |
| `thalach` | Max heart rate achieved (bpm) | Exercise stress tolerance capability |
| `exang` | Exercise-induced angina (1 = yes) | Ischemic coronary response during exertion |
| `oldpeak` | ST depression induced by exercise | Myocardial ischemia depth during exertion |
| `slope` | Peak exercise ST segment slope | Cardiac repolarization kinetics |
| `ca` | Major vessels colored by fluoroscopy (0–3) | Coronary artery calcification count |
| `thal` | Thallium scintigraphy stress test | Reversible vs. fixed perfusion defects |

---

## 3. Clinical Feature Engineering

To capture physiological cardiac relationships, we engineered domain-specific metrics:
1. **Expected Max Heart Rate (`max_hr_expected`)**: Based on the established Haskell & Fox cardiology formula: $	ext{HR}_{	ext{max}} = 220 - 	ext{age}$.
2. **Heart Rate Reserve Ratio (`hr_reserve_ratio`)**: Peak achieved heart rate divided by expected maximum ($	ext{thalach} / 	ext{max\_hr\_expected}$).
3. **Blood Pressure to Cholesterol Ratio (`bp_chol_ratio`)**: Interaction between vascular pressure and lipid density.
4. **ST Slope Interaction (`st_slope_interaction`)**: $	ext{oldpeak} 	imes (	ext{slope} + 1)$ capturing exercise-induced ischemia magnitude.

---

## 4. Machine Learning Pipeline Architecture

```text
               Patient Clinical Records (303 rows)
                                │
                                ▼
  ┌─────────────────────────────────────────────────────────────┐
  │ 1. Cardiology Feature Engineering                           │
  │ • Computed HR Reserve Ratio (thalach / (220 - age))         │
  │ • Computed BP-Cholesterol & ST-Slope interaction terms       │
  └─────────────────────────────┬───────────────────────────────┘
                                │
                                ▼
  ┌─────────────────────────────────────────────────────────────┐
  │ 2. Preprocessing & Scaling (StandardScaler)                 │
  │ • Normalized physiological attributes (trestbps, chol, etc.)│
  │ • Fit strictly on train split to prevent data leakage       │
  └─────────────────────────────┬───────────────────────────────┘
                                │
                                ▼
  ┌─────────────────────────────────────────────────────────────┐
  │ 3. Stratified Train / Test Split                            │
  │ • 80% Training Set, 20% Held-Out Testing Set                │
  └─────────────────────────────┬───────────────────────────────┘
                                │
                                ▼
  ┌─────────────────────────────────────────────────────────────┐
  │ 4. Model Benchmarking & Clinical Evaluation                 │
  │ • Support Vector Machine (RBF Kernel)                       │
  │ • Random Forest Classifier                                  │
  │ • Gradient Boosting Classifier                              │
  │ • Logistic Regression                                       │
  └─────────────────────────────┬───────────────────────────────┘
                                │
                                ▼
  ┌─────────────────────────────────────────────────────────────┐
  │ 5. Selected Diagnostic Winner: SVM (RBF Kernel)             │
  │ • Accuracy: 90.16% | Sensitivity: 93.94% | ROC-AUC: 91.56% │
  │ • Exported: models/heart_model.pkl & preprocessor.pkl       │
  └─────────────────────────────────────────────────────────────┘
```

---

## 5. Model Evaluation Benchmarks

Evaluated on the identical 20% held-out test split:

| Algorithm | Diagnostic Accuracy | Clinical Sensitivity (Recall) | Precision | F1-Score | ROC-AUC |
|---|---|---|---|---|---|
| **Support Vector Machine (RBF) ★** | **90.16%** | **93.94%** | **88.57%** | **91.18%** | **91.56%** |
| **Random Forest Classifier** | **86.89%** | **90.91%** | **85.71%** | **88.24%** | **90.80%** |
| **Gradient Boosting** | **81.97%** | **87.88%** | **80.56%** | **84.06%** | **89.61%** |
| **Logistic Regression** | **81.97%** | **87.88%** | **80.56%** | **84.06%** | **89.39%** |

### Why Sensitivity (Recall) is the Critical Clinical Metric
* **False Negative (Missed Diagnosis)**: A cardiac patient is diagnosed as healthy, receives no treatment, and faces imminent risk of fatal myocardial infarction.
* **False Positive (Overdiagnosis)**: A healthy patient is flagged for follow-up, undergoing non-invasive secondary testing (echocardiogram) before confirmation.
* **Clinical Imperative**: Sensitivity must be maximized to prevent fatal missed diagnoses. Our model achieves **93.94% Sensitivity**, successfully identifying **31 out of 33 test disease patients**!

---

## 6. Top Clinical Predictive Risk Drivers

Gini feature importance analysis confirms key cardiological indicators:
1. **Chest Pain Type (`cp`)**: Non-anginal and atypical chest pain correlate strongly with underlying ischemia.
2. **Major Vessels Colored (`ca`)**: Fluoroscopy vessel occlusion is directly indicative of coronary artery disease.
3. **Thalassemia Defect (`thal`)**: Perfusion defects during stress testing signal compromised blood delivery.
4. **Heart Rate Reserve (`hr_reserve_ratio`)**: Inability to achieve normal heart rate limits during exertion is an early cardiac warning signal.

---

## 7. Repository Structure

```text
heart-disease-prediction/
├── data/
│   └── heart.csv                       # Clinical Cleveland dataset (303 records)
├── models/
│   ├── heart_model.pkl                 # Trained SVM Diagnostic Model (~150 KB)
│   └── preprocessor.pkl                # Fitted StandardScaler (~1.5 KB)
├── notebooks/
│   └── cardiovascular_risk_prediction.ipynb # 11-step interactive notebook with plots
├── train_and_export.py                 # Standalone clinical training script
├── requirements.txt                    # Project dependencies
├── .gitignore                          # Clean Git ignore rules
└── README.md                           # Clinical documentation & interview guide
```

---

## 8. Quickstart Guide

### 1. Setup Environment
```bash
git clone https://github.com/<your-username>/heart-disease-prediction.git
cd heart-disease-prediction

python -m venv venv
# Windows:
.\venv\Scripts\activate
# Mac/Linux:
source venv/bin/activate

pip install -r requirements.txt
```

### 2. Run the Clinical Pipeline
```bash
python train_and_export.py
```

### 3. Open the Interactive Notebook
```bash
jupyter notebook notebooks/cardiovascular_risk_prediction.ipynb
```

---

## 9. Real-Time Patient Diagnostic Inference

```python
import joblib
import pandas as pd

# Load saved preprocessor and model
scaler = joblib.load("models/preprocessor.pkl")
model = joblib.load("models/heart_model.pkl")

# Patient test record
patient = pd.DataFrame([{
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

scaled_features = scaler.transform(patient)
diagnosis = model.predict(scaled_features)[0]
confidence = model.predict_proba(scaled_features)[0][1]

print(f"Diagnostic Result: {'POSITIVE (Cardiac Risk Detected)' if diagnosis == 1 else 'NEGATIVE (Normal)'}")
print(f"Risk Confidence: {confidence*100:.1f}%")
# Output: POSITIVE (Cardiac Risk Detected), Confidence: 86.4%
```

---

## 10. Technical & Clinical Interview Q&As

### Q1: Why is Recall (Sensitivity) preferred over Accuracy in Healthcare ML?
> **Answer**: In clinical diagnostics, the cost of a **False Negative** (failing to diagnose a sick patient) is potentially fatal. A **False Positive** merely leads to confirmatory testing. Therefore, while our model achieves **90.16% accuracy**, we specifically validated that clinical **Sensitivity is 93.94%** and **ROC-AUC is 91.56%**.

### Q2: How did you select the Support Vector Machine (SVM) over Random Forest?
> **Answer**: For clinical tabular datasets with complex non-linear boundary thresholds and normalized numerical measurements, the SVM with a Radial Basis Function (RBF) kernel effectively constructs high-dimensional maximum-margin hyperplanes, delivering superior generalization (**90.16% accuracy vs. 86.89%**).

### Q3: What clinical feature engineering did you introduce?
> **Answer**: We integrated established cardiology physiological metrics: the **Expected Maximum Heart Rate** ($220 - 	ext{age}$), the **Heart Rate Reserve Utilization** ($	ext{thalach} / 	ext{max\_hr}$), the **Blood Pressure to Cholesterol ratio**, and the **ST Depression-Slope interaction** to represent physiological exercise stress response.
