# 🎓 Student Performance Prediction & Analysis Dashboard

An end-to-end Machine Learning web application and analytics platform built with **Python**, **Streamlit**, **Scikit-Learn**, and **Plotly**. The system enables educational stakeholders (teachers, counselors, academic administrators) to analyze student demographics, uncover behavioral drivers of performance, benchmark predictive algorithms, and proactively intervene with at-risk students before final examinations.

---

## 🌟 Key Features

### 1. 📊 Exploratory Data Analysis (EDA)
- **Score Distributions & Pass Thresholds**: Interactive histograms and donut charts visualizing the Portugal standard grading scale (0-20, with 10 as pass cutoff).
- **Interactive Correlation Matrix**: Heatmap examining relationships across study hours, attendance, past failures, lifestyle factors (alcohol intake, leisure time), and final grades.
- **Academic & Social Driver Breakdowns**: Deep-dives into parental education, higher education aspirations, and absences.

### 2. 🤖 Machine Learning Model Studio & Benchmarking
- **Regression Suite (Grade Estimation)**:
  - *Ridge Regression*, *Random Forest Regressor*, *Gradient Boosting Regressor*
  - Metrics: $R^2$ Score, RMSE, MAE
- **Classification Suite (Pass / Fail Prediction)**:
  - *Logistic Regression*, *Random Forest Classifier*, *Gradient Boosting Classifier*
  - Metrics: Accuracy, Precision, Recall, F1 Score, ROC-AUC
- **Model Explainability & Feature Importance**: Identifies the primary drivers of student performance (past grades $G1$/$G2$, course failures, absences, study time).
- **Residual & Error Analysis**: Actual vs. Predicted scatter plot with ideal $y=x$ calibration line.

### 3. 🎯 Real-Time Student Predictor
- Input any student's academic history, study habits, and lifestyle indicators.
- Instant model inference:
  - **Predicted Final Grade** ($G3$ on 0-20 scale and converted percentage).
  - **Pass Probability** (%) with confidence color coding.
  - **Academic Risk Level** (`Low`, `Moderate`, `High`).
- **Personalized Recommendations**: Automatically generated action items (attendance alerts, study schedule optimization, peer tutoring suggestions).

### 4. 🔮 "What-If" Scenario Simulator
- Test hypothetical interventions dynamically:
  - *"What if study time increases from 2 hours to 5-10 hours?"*
  - *"What if school absences are reduced by 70%?"*
  - *"What if school educational support is provided?"*
- Visual delta comparison showing score lift and risk mitigation.

### 5. 🚨 Early Warning & At-Risk Intervention Hub
- Dedicated dashboard for educators to monitor students in the danger zone ($G3 < 10$ or high failure history).
- Filterable student roster with direct **Export to CSV** for parent-teacher conferences and academic board reviews.

### 6. 📁 Batch Inference & Custom Data Upload
- Upload your own student CSV dataset (semicolon or comma-separated).
- Run automated pipeline inference on every row.
- Download the enriched CSV with predicted grades, pass probabilities, and risk flags.

---

## 📂 Project Architecture

```
student-performance-dashboard/
├── data/
│   ├── student-mat.csv          # Mathematics course benchmark (395 records)
│   ├── student-por.csv          # Portuguese course benchmark (649 records)
│   └── student_data_loader.py   # Dataset fetcher & benchmark generator
├── src/
│   ├── __init__.py
│   ├── data_processing.py       # Cleaning, feature engineering, encoding & scaling
│   ├── ml_models.py             # Regression & classification model training & benchmarking
│   ├── predictor.py             # Single & batch inference pipeline + recommendations
│   └── utils.py                 # Metric cards, custom styling, Plotly dark theme
├── app.py                       # Main Streamlit web dashboard application
├── requirements.txt             # Python dependencies
└── README.md                    # Project documentation
```

---

## 🚀 Quick Start Guide

### 1. Prerequisites
Ensure you have **Python 3.9+** installed on your system.

### 2. Navigate to Project Directory
```bash
cd /Users/yugrupareliya/.gemini/antigravity/scratch/student-performance-dashboard
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Launch the Dashboard
```bash
streamlit run app.py
```
The application will open in your default browser at `http://localhost:8501`.

---

## 📊 Dataset Benchmark Details

The benchmark data uses the authentic **UCI Machine Learning Repository: Student Performance Dataset** (Cortez and Silva, 2008), tracking high school students across secondary schools:
- **Schools**: Gabriel Pereira (`GP`) and Mousinho da Silveira (`MS`).
- **Features**: 33 attributes spanning demographic, social, family, and school-related variables.
- **Target**: Final exam score $G3$ (continuous 0-20 scale) and binary status (Pass $\ge 10$, Fail $< 10$).

---

## 🛠️ Tech Stack
- **Framework**: Streamlit
- **Machine Learning**: Scikit-Learn
- **Data Manipulation**: Pandas, NumPy
- **Visualizations**: Plotly Express, Plotly Graph Objects, Seaborn
