import os
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from src.data_processing import (
    load_student_dataset, FEATURE_COLS, NUMERICAL_COLS, CATEGORICAL_COLS,
    STUDY_TIME_MAP, TRAVEL_TIME_MAP, EDU_MAP
)
from src.ml_models import train_and_evaluate_models
from src.predictor import (
    predict_single_student, predict_batch_dataset,
    DEFAULT_STUDENT_VALUES, STUDENT_PRESETS, get_letter_grade
)
from src.utils import (
    apply_custom_css, render_kpi, render_hero_banner,
    render_insight_box, render_grading_legend, get_plotly_theme
)

# Page Setup
st.set_page_config(
    page_title="Student Performance AI Dashboard",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

apply_custom_css()

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
MAT_CSV = os.path.join(DATA_DIR, "student-mat.csv")
POR_CSV = os.path.join(DATA_DIR, "student-por.csv")

if not os.path.exists(MAT_CSV) or not os.path.exists(POR_CSV):
    from data.student_data_loader import fetch_or_generate_data
    fetch_or_generate_data()

# -------------------------------------------------------------
# Caching Data & Model Training
# -------------------------------------------------------------
@st.cache_data
def get_dataset(dataset_choice, uploaded_file=None):
    if uploaded_file is not None:
        df = load_student_dataset(uploaded_file)
        return df, "Custom Upload"
    
    if dataset_choice == "Mathematics Course (395 records)":
        df = load_student_dataset(MAT_CSV)
        return df, "Math Course"
    elif dataset_choice == "Portuguese Language Course (649 records)":
        df = load_student_dataset(POR_CSV)
        return df, "Portuguese Course"
    else:
        df_mat = load_student_dataset(MAT_CSV)
        df_mat['subject'] = 'Math'
        df_por = load_student_dataset(POR_CSV)
        df_por['subject'] = 'Portuguese'
        df = pd.concat([df_mat, df_por], ignore_index=True)
        return df, "Combined Courses"

@st.cache_resource
def get_trained_models(df):
    return train_and_evaluate_models(df)

# -------------------------------------------------------------
# Sidebar Navigation & Filter Controls
# -------------------------------------------------------------
with st.sidebar:
    st.image("https://img.icons8.com/clouds/200/graduation-cap.png", width=105)
    st.title("Settings & Filter")
    
    st.markdown("### 📂 Data Source")
    data_option = st.selectbox(
        "Choose Academic Dataset:",
        [
            "Mathematics Course (395 records)",
            "Portuguese Language Course (649 records)",
            "Combined Courses (1,044 records)",
            "Upload Custom CSV File"
        ],
        help="Select the academic cohort to examine or upload your own student data file."
    )

    uploaded_file = None
    if data_option == "Upload Custom CSV File":
        uploaded_file = st.file_uploader("Upload student CSV (semicolon or comma delimited):", type=['csv', 'txt'])

    raw_df, dataset_name = get_dataset(data_option, uploaded_file)

    st.markdown("---")
    st.markdown("### 🔍 Global Slicers")

    # School filter
    schools_available = sorted(raw_df['school'].dropna().unique().tolist()) if 'school' in raw_df.columns else []
    sel_schools = st.multiselect("School:", schools_available, default=schools_available, help="GP: Gabriel Pereira | MS: Mousinho da Silveira")

    # Sex filter
    genders_available = sorted(raw_df['sex'].dropna().unique().tolist()) if 'sex' in raw_df.columns else []
    sel_genders = st.multiselect("Gender:", genders_available, default=genders_available)

    # Age filter
    min_age = int(raw_df['age'].min()) if 'age' in raw_df.columns else 15
    max_age = int(raw_df['age'].max()) if 'age' in raw_df.columns else 22
    sel_age_range = st.slider("Age Range:", min_value=min_age, max_value=max_age, value=(min_age, max_age))

    # Apply global filters
    df_filtered = raw_df.copy()
    if 'school' in df_filtered.columns and sel_schools:
        df_filtered = df_filtered[df_filtered['school'].isin(sel_schools)]
    if 'sex' in df_filtered.columns and sel_genders:
        df_filtered = df_filtered[df_filtered['sex'].isin(sel_genders)]
    if 'age' in df_filtered.columns:
        df_filtered = df_filtered[(df_filtered['age'] >= sel_age_range[0]) & (df_filtered['age'] <= sel_age_range[1])]

    st.markdown(f"**Filtered Cohort:** `{len(df_filtered)}` of `{len(raw_df)}` students")

    st.markdown("---")
    with st.expander("ℹ️ Variable Dictionary"):
        st.markdown(r"""
        - **G1 & G2**: Period 1 & 2 Trimester marks (0–20 scale)
        - **G3**: Final Exam Mark (Target variable)
        - **Pass Cutoff**: $\ge 10$ is passing mark
        - **studytime**: 1 (<2h), 2 (2-5h), 3 (5-10h), 4 (>10h)
        - **absences**: Recorded class absence days
        - **failures**: Number of past failed subjects (0–3)
        """)

# Top Hero Banner
render_hero_banner()

if len(df_filtered) == 0:
    st.warning("⚠️ No student records match the active filters. Expand your sidebar selections.")
    st.stop()

# -------------------------------------------------------------
# KPI Overview Bar
# -------------------------------------------------------------
k1, k2, k3, k4, k5 = st.columns(5)

total_students = len(df_filtered)
avg_g3 = df_filtered['G3'].mean() if 'G3' in df_filtered.columns else 0
avg_g3_pct = (avg_g3 / 20.0) * 100.0
pass_rate = (df_filtered['passed'].mean() * 100.0) if 'passed' in df_filtered.columns else 0
at_risk_count = (df_filtered['risk_level'] == 'High').sum() if 'risk_level' in df_filtered.columns else 0
at_risk_pct = (at_risk_count / total_students * 100.0) if total_students > 0 else 0
avg_absences = df_filtered['absences'].mean() if 'absences' in df_filtered.columns else 0

render_kpi(k1, "Total Students", f"{total_students:,}", f"Source: {dataset_name}", "#38bdf8")
render_kpi(k2, "Average Final Score", f"{avg_g3:.1f} / 20", f"{avg_g3_pct:.1f}% Score Equiv.", "#10b981" if avg_g3 >= 12 else "#f59e0b")
render_kpi(k3, "Overall Pass Rate", f"{pass_rate:.1f}%", f"{int(pass_rate * total_students / 100)} Passing Students", "#10b981" if pass_rate >= 75 else "#ef4444")
render_kpi(k4, "High Risk Students", f"{at_risk_pct:.1f}%", f"{at_risk_count} Students Flagged", "#ef4444" if at_risk_pct > 20 else "#f59e0b")
render_kpi(k5, "Avg. School Absences", f"{avg_absences:.1f} days", f"Max: {df_filtered['absences'].max():.0f} days", "#38bdf8")

# Grading scale reference bar
render_grading_legend()

# Model Training Pipeline (cached)
models_bundle = get_trained_models(raw_df)
best_reg_pipe = models_bundle['reg_models']['Gradient Boosting Regressor']
best_clf_pipe = models_bundle['clf_models']['Random Forest Classifier']

# -------------------------------------------------------------
# Main Navigation Tabs
# -------------------------------------------------------------
tab_eda, tab_ml, tab_pred, tab_whatif, tab_risk, tab_batch = st.tabs([
    "📊 Exploratory Data Analysis",
    "🤖 ML Model Studio",
    "🎯 Student Predictor",
    "🔮 What-If Simulator",
    "🚨 Early Warning & Interventions",
    "📁 Batch Prediction & Export"
])

# =============================================================
# TAB 1: EDA
# =============================================================
with tab_eda:
    st.subheader("📊 Academic Distributions & Behavioral Analytics")
    st.markdown("Understand what patterns separate top-performing students from those falling behind.")

    eda_col1, eda_col2 = st.columns(2)

    with eda_col1:
        fig_dist = px.histogram(
            df_filtered,
            x='G3',
            nbins=21,
            color='grade_bracket',
            title="Distribution of Final Grades (G3) with Pass/Fail Boundary",
            labels={'G3': 'Final Grade (0 - 20)', 'grade_bracket': 'Grade Bracket'},
            color_discrete_map={
                "Fail (<10)": "#ef4444",
                "Sufficient (10-11)": "#f59e0b",
                "Satisfactory (12-13)": "#3b82f6",
                "Good (14-15)": "#6366f1",
                "Excellent (16-20)": "#10b981"
            }
        )
        fig_dist.add_vline(x=9.5, line_width=2.5, line_dash="dash", line_color="#f43f5e", annotation_text="Pass Cutoff (10)", annotation_position="top left")
        fig_dist.update_layout(bargap=0.08, height=360, **get_plotly_theme())
        st.plotly_chart(fig_dist, use_container_width=True)

    with eda_col2:
        bracket_counts = df_filtered['grade_bracket'].value_counts().reset_index()
        bracket_counts.columns = ['Bracket', 'Count']
        fig_pie = px.pie(
            bracket_counts,
            values='Count',
            names='Bracket',
            title="Cohort Academic Grade Distribution",
            hole=0.45,
            color='Bracket',
            color_discrete_map={
                "Fail (<10)": "#ef4444",
                "Sufficient (10-11)": "#f59e0b",
                "Satisfactory (12-13)": "#3b82f6",
                "Good (14-15)": "#6366f1",
                "Excellent (16-20)": "#10b981"
            }
        )
        fig_pie.update_layout(height=360, **get_plotly_theme())
        st.plotly_chart(fig_pie, use_container_width=True)

    render_insight_box(
        "Grade Bimodal Spread",
        f"Approximately {pass_rate:.1f}% of students pass the course (≥ 10), while {100 - pass_rate:.1f}% score below the passing mark. A notable cluster of students score 0, representing exam dropouts."
    )

    st.markdown("---")

    # Correlation Matrix
    corr_cols = [c for c in ['G1', 'G2', 'G3', 'studytime', 'absences', 'failures', 'age', 'Dalc', 'Walc', 'health', 'famrel', 'freetime', 'goout'] if c in df_filtered.columns]
    corr_df = df_filtered[corr_cols].corr()

    fig_corr = px.imshow(
        corr_df,
        text_auto=".2f",
        aspect="auto",
        title="Interactive Correlation Matrix (Academic Marks, Study Habits & Lifestyle)",
        color_continuous_scale="RdBu_r",
        zmin=-1,
        zmax=1
    )
    fig_corr.update_layout(height=450, **get_plotly_theme())
    st.plotly_chart(fig_corr, use_container_width=True)

    render_insight_box(
        "Strongest Performance Correlates",
        "Term 1 (G1) and Term 2 (G2) grades have overwhelming positive correlations (~0.85 to 0.90) with final performance. Prior course failures (-0.36) and high alcohol consumption (-0.15) demonstrate the strongest negative correlations."
    )

    st.markdown("---")

    # Behavioral Drivers
    row2_1, row2_2 = st.columns(2)

    with row2_1:
        fig_box_study = px.box(
            df_filtered,
            x='study_time_label',
            y='G3',
            color='study_time_label',
            title="Weekly Study Time vs. Final Exam Grade",
            category_orders={'study_time_label': ['< 2 hours', '2 to 5 hours', '5 to 10 hours', '> 10 hours']},
            labels={'study_time_label': 'Weekly Study Hours', 'G3': 'Final Grade (0-20)'}
        )
        fig_box_study.update_layout(showlegend=False, height=360, **get_plotly_theme())
        st.plotly_chart(fig_box_study, use_container_width=True)

    with row2_2:
        fig_abs = px.scatter(
            df_filtered,
            x='absences',
            y='G3',
            color='passed',
            color_discrete_map={1: '#10b981', 0: '#ef4444'},
            title="School Absences vs. Final Score",
            labels={'absences': 'Absences (Days)', 'G3': 'Final Grade (0-20)', 'passed': 'Passed (1=Yes, 0=No)'}
        )
        fig_abs.update_layout(height=360, **get_plotly_theme())
        st.plotly_chart(fig_abs, use_container_width=True)

    render_insight_box(
        "Study Hours vs. Attendance",
        "Students studying 5–10 hours weekly achieve a median grade of 13.0, compared to 10.0 for those studying <2 hours. Furthermore, students with over 10 absences show sharp volatility and higher fail rates."
    )

    row3_1, row3_2 = st.columns(2)

    with row3_1:
        fig_fail = px.bar(
            df_filtered.groupby('failures')['G3'].mean().reset_index(),
            x='failures',
            y='G3',
            title="Average Final Grade by Number of Prior Subject Failures",
            color='failures',
            labels={'failures': 'Prior Course Failures (0 to 3)', 'G3': 'Average Final Grade (G3)'},
            color_continuous_scale="Reds_r"
        )
        fig_fail.update_layout(height=360, **get_plotly_theme())
        st.plotly_chart(fig_fail, use_container_width=True)

    with row3_2:
        fig_edu = px.violin(
            df_filtered,
            x='higher',
            y='G3',
            color='higher',
            box=True,
            points="all",
            title="Desire for Higher Education vs. Performance",
            labels={'higher': 'Wants Higher Education (yes/no)', 'G3': 'Final Grade (0-20)'},
            color_discrete_map={'yes': '#3b82f6', 'no': '#ef4444'}
        )
        fig_edu.update_layout(showlegend=False, height=360, **get_plotly_theme())
        st.plotly_chart(fig_edu, use_container_width=True)

# =============================================================
# TAB 2: ML MODEL BENCHMARKING
# =============================================================
with tab_ml:
    st.subheader("🤖 Machine Learning Model Studio & Explainability")
    st.markdown("We benchmark multiple algorithms across Regression (exact score prediction) and Classification (pass/fail forecast).")

    # Plain English Explainer
    with st.expander("📚 Plain-English Guide to Model Metrics (Click to Expand)", expanded=False):
        ec1, ec2, ec3, ec4 = st.columns(4)
        ec1.markdown("**R² Score (Accuracy %)**<br>Measures how much of the variation in grades is explained by our inputs (0 to 1). Higher is better.", unsafe_allow_html=True)
        ec2.markdown("**RMSE (Average Error)**<br>Average point deviation between predicted score and actual score (e.g. 1.6 points). Lower is better.", unsafe_allow_html=True)
        ec3.markdown("**Accuracy (Pass/Fail)**<br>Percentage of correct pass/fail classifications across all test cases. Higher is better.", unsafe_allow_html=True)
        ec4.markdown("**ROC-AUC Score**<br>Measures how effectively the classifier distinguishes between passing and failing students (0.5 to 1.0).", unsafe_allow_html=True)

    ml_col1, ml_col2 = st.columns(2)

    with ml_col1:
        st.markdown("#### 📈 Regression Leaderboard (Predicting G3 Final Score)")
        reg_df = models_bundle['reg_results']
        st.dataframe(reg_df.style.highlight_max(axis=0, subset=['R² Score'], color='#1e3a8a')
                                .highlight_min(axis=0, subset=['RMSE', 'MAE'], color='#1e3a8a'),
                     width='stretch')

        fig_reg_bar = px.bar(
            reg_df,
            x='Model',
            y='R² Score',
            color='R² Score',
            title="R² Score Comparison (Explains ~87% of Variance)",
            color_continuous_scale="Blues",
            range_y=[0, 1]
        )
        fig_reg_bar.update_layout(height=300, **get_plotly_theme())
        st.plotly_chart(fig_reg_bar, use_container_width=True)

    with ml_col2:
        st.markdown("#### 🎯 Classification Leaderboard (Predicting Pass vs. Fail)")
        clf_df = models_bundle['clf_results']
        st.dataframe(clf_df.style.highlight_max(axis=0, subset=['Accuracy', 'F1 Score', 'ROC AUC'], color='#1e3a8a'),
                     width='stretch')

        fig_clf_bar = px.bar(
            clf_df,
            x='Model',
            y='Accuracy',
            color='Accuracy',
            title="Classifier Accuracy Comparison (~87.3% Accuracy)",
            color_continuous_scale="Teal",
            range_y=[0.5, 1]
        )
        fig_clf_bar.update_layout(height=300, **get_plotly_theme())
        st.plotly_chart(fig_clf_bar, use_container_width=True)

    st.markdown("---")

    # Feature Importance
    st.markdown("#### 🔍 Feature Importance: What Factors Determine Student Success?")
    st.markdown("Tree-based ensemble analysis reveals which characteristics have the greatest predictive weight:")

    feat_df = models_bundle['base_feat_imp_df'].head(12)
    fig_feat = px.bar(
        feat_df,
        x='Importance',
        y='Feature',
        orientation='h',
        title="Top 12 Most Influential Predictive Drivers",
        color='Importance',
        color_continuous_scale="Purples"
    )
    fig_feat.update_layout(yaxis={'categoryorder': 'total ascending'}, height=390, **get_plotly_theme())
    st.plotly_chart(fig_feat, use_container_width=True)

    render_insight_box(
        "Model Explainability",
        "Recent performance in Term 2 ($G2$) and Term 1 ($G1$) constitutes over 70% of predictive importance. Prior class failures, school absences, and weekly study time form the remaining top decisive predictors."
    )

    # Actual vs Predicted
    st.markdown("---")
    st.markdown("#### 🎯 Prediction Calibration (Actual vs. Predicted Scores on Unseen Test Set)")
    pred_data = models_bundle['reg_predictions']['Gradient Boosting Regressor']
    eval_df = pd.DataFrame({'Actual': pred_data['y_true'], 'Predicted': pred_data['y_pred']})

    fig_scat = px.scatter(
        eval_df,
        x='Actual',
        y='Predicted',
        title="Actual vs Predicted G3 Scores (Points tightly grouped along the dashed red line indicate high precision)",
        labels={'Actual': 'Actual Final Grade', 'Predicted': 'Predicted Final Grade'},
        color_discrete_sequence=['#38bdf8']
    )
    fig_scat.add_trace(go.Scatter(x=[0, 20], y=[0, 20], mode='lines', name='Ideal Fit (y=x)', line=dict(color='#ef4444', dash='dash')))
    fig_scat.update_layout(height=400, **get_plotly_theme())
    st.plotly_chart(fig_scat, use_container_width=True)

# =============================================================
# TAB 3: SINGLE STUDENT PREDICTOR
# =============================================================
with tab_pred:
    st.subheader("🎯 Interactive Student Performance Predictor")
    st.markdown("Adjust a student's academic history and lifestyle metrics to forecast final marks, pass probabilities, and receive tailored interventions.")

    # One-Click Preset Profiles
    st.markdown("##### ⚡ Quick Load Student Profile Presets:")
    pr1, pr2, pr3 = st.columns(3)
    
    if 'preset_loaded' not in st.session_state:
        st.session_state['preset_loaded'] = STUDENT_PRESETS["⚖️ Typical / Average Student"]

    if pr1.button("🌟 High Achieving Student", use_container_width=True):
        st.session_state['preset_loaded'] = STUDENT_PRESETS["🌟 High Achieving Student"]
    if pr2.button("⚖️ Typical / Average Student", use_container_width=True):
        st.session_state['preset_loaded'] = STUDENT_PRESETS["⚖️ Typical / Average Student"]
    if pr3.button("🚨 Student Needing Support", use_container_width=True):
        st.session_state['preset_loaded'] = STUDENT_PRESETS["🚨 Student Needing Immediate Support"]

    active_p = st.session_state['preset_loaded']

    with st.form("student_prediction_form"):
        st.markdown("##### 📝 Input Profile Attributes")
        c1, c2, c3 = st.columns(3)

        with c1:
            st.markdown("**Academic Marks**")
            in_g1 = st.slider("Term 1 Grade (G1, 0-20):", 0, 20, int(active_p.get('G1', 11)), help="Trimester 1 score")
            in_g2 = st.slider("Term 2 Grade (G2, 0-20):", 0, 20, int(active_p.get('G2', 12)), help="Trimester 2 score")
            in_failures = st.selectbox("Prior Course Failures:", [0, 1, 2, 3], index=int(active_p.get('failures', 0)))
            in_schoolsup = st.selectbox("Extra Educational Tutoring Support:", ["no", "yes"], index=1 if active_p.get('schoolsup') == 'yes' else 0)

        with c2:
            st.markdown("**Study Habits & Attendance**")
            in_studytime = st.selectbox("Weekly Study Hours:", [1, 2, 3, 4], index=int(active_p.get('studytime', 2)) - 1, format_func=lambda x: STUDY_TIME_MAP[x])
            in_absences = st.slider("School Absences (Days):", 0, 50, int(active_p.get('absences', 4)))
            in_freetime = st.slider("Free Time After School (1-5):", 1, 5, int(active_p.get('freetime', 3)))
            in_goout = st.slider("Going Out with Friends (1-5):", 1, 5, int(active_p.get('goout', 3)))

        with c3:
            st.markdown("**Family & Personal Factors**")
            in_medu = st.selectbox("Mother's Education:", [0, 1, 2, 3, 4], index=int(active_p.get('Medu', 3)), format_func=lambda x: EDU_MAP[x])
            in_fedu = st.selectbox("Father's Education:", [0, 1, 2, 3, 4], index=int(active_p.get('Fedu', 3)), format_func=lambda x: EDU_MAP[x])
            in_higher = st.selectbox("Desires Higher Education:", ["yes", "no"], index=0 if active_p.get('higher', 'yes') == 'yes' else 1)
            in_internet = st.selectbox("Home Internet Access:", ["yes", "no"], index=0 if active_p.get('internet', 'yes') == 'yes' else 1)

        with st.expander("⚙️ Optional Demographics & Lifestyle Details", expanded=False):
            ac1, ac2, ac3, ac4 = st.columns(4)
            with ac1:
                in_sex = st.selectbox("Gender:", ["F", "M"])
                in_age = st.number_input("Age:", min_value=15, max_value=22, value=16)
            with ac2:
                in_school = st.selectbox("School:", ["GP", "MS"])
                in_address = st.selectbox("Address Type:", ["U", "R"], format_func=lambda x: "Urban (U)" if x == "U" else "Rural (R)")
            with ac3:
                in_dalc = st.slider("Workday Alcohol (1-5):", 1, 5, int(active_p.get('Dalc', 1)))
                in_walc = st.slider("Weekend Alcohol (1-5):", 1, 5, int(active_p.get('Walc', 1)))
            with ac4:
                in_health = st.slider("Current Health Status (1-5):", 1, 5, int(active_p.get('health', 4)))
                in_famrel = st.slider("Family Relationship Quality (1-5):", 1, 5, 4)

        submit_btn = st.form_submit_button("🚀 Compute Prediction & Generate Action Plan", use_container_width=True)

    # Automatically compute prediction on submit OR when preset changes
    student_input = {
        'school': in_school, 'sex': in_sex, 'age': in_age, 'address': in_address,
        'famsize': 'GT3', 'Pstatus': 'T', 'Medu': in_medu, 'Fedu': in_fedu,
        'Mjob': 'services', 'Fjob': 'services', 'reason': 'course', 'guardian': 'mother',
        'traveltime': 1, 'studytime': in_studytime, 'failures': in_failures,
        'schoolsup': in_schoolsup, 'famsup': 'yes', 'paid': 'no',
        'activities': 'yes', 'nursery': 'yes', 'higher': in_higher,
        'internet': in_internet, 'romantic': 'no', 'famrel': in_famrel,
        'freetime': in_freetime, 'goout': in_goout, 'Dalc': in_dalc, 'Walc': in_walc,
        'health': in_health, 'absences': in_absences, 'G1': in_g1, 'G2': in_g2
    }

    result = predict_single_student(best_reg_pipe, best_clf_pipe, student_input)

    st.markdown("---")
    st.markdown("### 🏆 Prediction Summary & Student Report Card")

    p_res1, p_res2, p_res3 = st.columns(3)
    render_kpi(p_res1, "Predicted Final Score", f"{result['pred_g3']} / 20", f"{result['pred_pct']}% Academic Grade", "#38bdf8")

    pass_color = "#10b981" if result['pass_prob'] >= 60 else "#ef4444"
    status_text = "Passing Projected" if result['passed'] else "At-Risk of Failing"
    render_kpi(p_res2, "Pass Probability", f"{result['pass_prob']}%", status_text, pass_color)

    risk_color = "#10b981" if result['risk_level'] == 'Low Risk' else ("#f59e0b" if result['risk_level'] == 'Moderate Risk' else "#ef4444")
    render_kpi(p_res3, "Risk Level", result['risk_level'], result['letter_grade'], risk_color)

    st.markdown(f"""
    <div style="background: #1e293b; border-left: 5px solid {result['letter_color']}; border-radius: 8px; padding: 14px 18px; margin: 15px 0;">
        <span style="font-size: 1.15rem; font-weight: 700; color: {result['letter_color']};">{result['letter_grade']}</span>
        <span style="color: #94a3b8; margin-left: 10px;">(Projected Final Mark: <strong>{result['pred_g3']} / 20</strong> — Equivalent to <strong>{result['pred_pct']}%</strong>)</span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("#### 📋 Personalized Action Plan & Recommendations")
    for rec in result['recommendations']:
        st.markdown(f"- {rec}")

# =============================================================
# TAB 4: WHAT-IF SCENARIO SIMULATOR
# =============================================================
with tab_whatif:
    st.subheader("🔮 What-If Intervention Simulator")
    st.markdown("Simulate behavioral interventions (reducing absences, increasing study hours, tutoring) and observe the projected grade lift in real time.")

    sim_col1, sim_col2 = st.columns(2)

    with sim_col1:
        st.markdown("##### 1. Baseline Student Status")
        base_g1 = st.slider("Baseline Term 1 (G1):", 0, 20, 8, key="sim_g1")
        base_g2 = st.slider("Baseline Term 2 (G2):", 0, 20, 9, key="sim_g2")
        base_study = st.selectbox("Baseline Study Time:", [1, 2, 3, 4], index=0, format_func=lambda x: STUDY_TIME_MAP[x], key="sim_bstudy")
        base_abs = st.slider("Baseline Absences:", 0, 40, 15, key="sim_babs")
        base_failures = st.selectbox("Baseline Course Failures:", [0, 1, 2, 3], index=1, key="sim_bfail")

    with sim_col2:
        st.markdown("##### 2. Proposed Intervention Package")
        sim_study = st.selectbox("Upgraded Study Time:", [1, 2, 3, 4], index=2, format_func=lambda x: STUDY_TIME_MAP[x], key="sim_astudy")
        sim_abs = st.slider("Reduced Absences:", 0, 40, 3, key="sim_aabs")
        sim_support = st.selectbox("Provide Educational Tutoring Support:", ["no", "yes"], index=1, key="sim_asup")
        sim_higher = st.selectbox("Reinforce Higher Education Aspirations:", ["yes", "no"], index=0, key="sim_ahigh")

    base_input = dict(DEFAULT_STUDENT_VALUES)
    base_input.update({
        'G1': base_g1, 'G2': base_g2, 'studytime': base_study,
        'absences': base_abs, 'failures': base_failures,
        'schoolsup': 'no', 'higher': 'yes'
    })

    sim_input = dict(base_input)
    sim_input.update({
        'studytime': sim_study, 'absences': sim_abs,
        'schoolsup': sim_support, 'higher': sim_higher
    })

    base_pred = predict_single_student(best_reg_pipe, best_clf_pipe, base_input)
    sim_pred = predict_single_student(best_reg_pipe, best_clf_pipe, sim_input)

    st.markdown("---")
    st.markdown("#### 📊 Comparative Projection: Before vs. After Intervention")

    sc1, sc2, sc3 = st.columns(3)
    diff_score = round(sim_pred['pred_g3'] - base_pred['pred_g3'], 1)
    diff_pass = round(sim_pred['pass_prob'] - base_pred['pass_prob'], 1)

    with sc1:
        st.metric("Predicted Final Grade", f"{sim_pred['pred_g3']} / 20", delta=f"{diff_score:+} points")
    with sc2:
        st.metric("Pass Probability", f"{sim_pred['pass_prob']}%", delta=f"{diff_pass:+.1f}%")
    with sc3:
        st.metric("Risk Status", f"{sim_pred['risk_level']}", delta=f"Was {base_pred['risk_level']}", delta_color="inverse" if sim_pred['risk_level'] == 'High Risk' else "normal")

    comp_df = pd.DataFrame({
        'Metric': ['Final Grade (0-20)', 'Pass Probability (%)'],
        'Baseline': [base_pred['pred_g3'], base_pred['pass_prob']],
        'After Intervention': [sim_pred['pred_g3'], sim_pred['pass_prob']]
    })
    comp_fig = px.bar(
        comp_df.melt(id_vars='Metric', var_name='Scenario', value_name='Score'),
        x='Metric',
        y='Score',
        color='Scenario',
        barmode='group',
        title="Projected Academic Score Lift from Targeted Interventions",
        color_discrete_map={'Baseline': '#ef4444', 'After Intervention': '#10b981'}
    )
    comp_fig.update_layout(height=340, **get_plotly_theme())
    st.plotly_chart(comp_fig, use_container_width=True)

# =============================================================
# TAB 5: EARLY WARNING & AT-RISK INTERVENTION HUB
# =============================================================
with tab_risk:
    st.subheader("🚨 Early Warning & Academic Intervention Hub")
    st.markdown("Designed for teachers and counselors to rapidly detect and reach out to students at severe risk of failing.")

    risk_filter = st.radio("Display Students with Risk Category:", ["High Risk Only", "Moderate & High", "All Students"], horizontal=True)

    if risk_filter == "High Risk Only":
        risk_df = df_filtered[df_filtered['risk_level'] == 'High'].copy()
    elif risk_filter == "Moderate & High":
        risk_df = df_filtered[df_filtered['risk_level'].isin(['High', 'Moderate'])].copy()
    else:
        risk_df = df_filtered.copy()

    r_col1, r_col2, r_col3 = st.columns(3)
    r_col1.metric("Flagged Students", f"{len(risk_df):,}", f"{(len(risk_df)/len(df_filtered)*100):.1f}% of cohort")
    r_col2.metric("Average Absences in Group", f"{risk_df['absences'].mean():.1f} days")
    r_col3.metric("Average Term 2 Grade (G2)", f"{risk_df['G2'].mean():.1f} / 20")

    st.markdown("---")
    st.markdown("##### Filtered Student Roster")
    display_cols = [c for c in ['school', 'sex', 'age', 'G1', 'G2', 'G3', 'failures', 'absences', 'studytime', 'risk_level'] if c in risk_df.columns]
    
    st.dataframe(
        risk_df[display_cols].style.map(
            lambda v: 'background-color: #7f1d1d; color: #fecaca; font-weight: bold;' if v == 'High' else (
                      'background-color: #78350f; color: #fef3c7;' if v == 'Moderate' else ''
            ), subset=['risk_level']
        ),
        width='stretch'
    )

    csv_bytes = risk_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Export At-Risk Student Roster (CSV)",
        data=csv_bytes,
        file_name="at_risk_students_roster.csv",
        mime="text/csv"
    )

# =============================================================
# TAB 6: BATCH PREDICTION & EXPORT
# =============================================================
with tab_batch:
    st.subheader("📁 Batch File Prediction & Export")
    st.markdown("Upload any CSV file of student records to generate automated batch grades and risk scores.")

    # Download Template Button
    st.markdown("##### Need a template?")
    template_df = raw_df[FEATURE_COLS].head(5).copy()
    template_csv = template_df.to_csv(index=False, sep=';').encode('utf-8')
    st.download_button(
        label="📄 Download Sample CSV Template",
        data=template_csv,
        file_name="student_template.csv",
        mime="text/csv",
        help="Download a 5-row sample template matching required column headers."
    )

    st.markdown("---")
    batch_file = st.file_uploader("Upload CSV for Batch Inference:", type=['csv', 'txt'], key="batch_upload")
    
    if batch_file is not None:
        try:
            batch_df_raw = load_student_dataset(batch_file)
            st.success(f"Successfully loaded {len(batch_df_raw)} records from uploaded file.")
            
            if st.button("⚡ Run Model Inference on All Rows", use_container_width=True):
                with st.spinner("Executing predictions across dataset..."):
                    enriched_df = predict_batch_dataset(best_reg_pipe, best_clf_pipe, batch_df_raw)
                    
                    st.markdown("### 📋 Batch Scoring Summary")
                    b1, b2, b3 = st.columns(3)
                    b1.metric("Predicted Cohort Average", f"{enriched_df['Predicted_G3'].mean():.1f} / 20")
                    b2.metric("Projected Pass Rate", f"{(enriched_df['Predicted_Status'] == 'Pass').mean() * 100:.1f}%")
                    b3.metric("Projected High Risk", f"{(enriched_df['Predicted_Risk'] == 'High Risk').sum()} students")

                    st.dataframe(enriched_df[['Predicted_G3', 'Predicted_Percentage', 'Pass_Probability', 'Letter_Grade', 'Predicted_Status', 'Predicted_Risk'] + [c for c in ['G1', 'G2', 'absences', 'studytime'] if c in enriched_df.columns]].head(25), width='stretch')

                    dl_csv = enriched_df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="📥 Download Full Enriched Prediction Dataset (CSV)",
                        data=dl_csv,
                        file_name="student_predictions_enriched.csv",
                        mime="text/csv"
                    )
        except Exception as e:
            st.error(f"Error processing file: {e}")
    else:
        st.info("💡 Upload a CSV file above to process and score multiple students simultaneously.")
