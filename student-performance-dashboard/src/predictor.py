import numpy as np
import pandas as pd
from src.data_processing import FEATURE_COLS, STUDY_TIME_MAP

DEFAULT_STUDENT_VALUES = {
    'school': 'GP',
    'sex': 'F',
    'age': 16,
    'address': 'U',
    'famsize': 'GT3',
    'Pstatus': 'T',
    'Medu': 3,
    'Fedu': 3,
    'Mjob': 'services',
    'Fjob': 'services',
    'reason': 'course',
    'guardian': 'mother',
    'traveltime': 1,
    'studytime': 2,
    'failures': 0,
    'schoolsup': 'no',
    'famsup': 'yes',
    'paid': 'no',
    'activities': 'yes',
    'nursery': 'yes',
    'higher': 'yes',
    'internet': 'yes',
    'romantic': 'no',
    'famrel': 4,
    'freetime': 3,
    'goout': 3,
    'Dalc': 1,
    'Walc': 2,
    'health': 4,
    'absences': 4,
    'G1': 12,
    'G2': 12
}

STUDENT_PRESETS = {
    "🌟 High Achieving Student": {
        'G1': 16, 'G2': 17, 'studytime': 3, 'absences': 1, 'failures': 0,
        'schoolsup': 'no', 'famsup': 'yes', 'higher': 'yes', 'internet': 'yes',
        'Medu': 4, 'Fedu': 4, 'freetime': 3, 'goout': 2, 'Dalc': 1, 'Walc': 1, 'health': 5
    },
    "⚖️ Typical / Average Student": {
        'G1': 11, 'G2': 11, 'studytime': 2, 'absences': 4, 'failures': 0,
        'schoolsup': 'no', 'famsup': 'yes', 'higher': 'yes', 'internet': 'yes',
        'Medu': 2, 'Fedu': 2, 'freetime': 3, 'goout': 3, 'Dalc': 1, 'Walc': 2, 'health': 4
    },
    "🚨 Student Needing Immediate Support": {
        'G1': 8, 'G2': 7, 'studytime': 1, 'absences': 16, 'failures': 2,
        'schoolsup': 'no', 'famsup': 'no', 'higher': 'no', 'internet': 'no',
        'Medu': 1, 'Fedu': 1, 'freetime': 4, 'goout': 5, 'Dalc': 3, 'Walc': 4, 'health': 2
    }
}

def get_letter_grade(score):
    if score >= 16.0:
        return "Grade A (Excellent)", "#10b981"
    elif score >= 14.0:
        return "Grade B (Good)", "#3b82f6"
    elif score >= 12.0:
        return "Grade C (Satisfactory)", "#6366f1"
    elif score >= 10.0:
        return "Grade D (Passing)", "#f59e0b"
    else:
        return "Grade F (Failing / At-Risk)", "#ef4444"

def predict_single_student(reg_pipe, clf_pipe, student_dict):
    """
    Infers final grade G3, pass probability, risk level, letter grade, and recommendations.
    """
    row_data = {}
    for col in FEATURE_COLS:
        row_data[col] = student_dict.get(col, DEFAULT_STUDENT_VALUES.get(col))

    df_input = pd.DataFrame([row_data])

    # Continuous G3 prediction
    pred_g3 = float(reg_pipe.predict(df_input)[0])
    pred_g3 = float(np.clip(round(pred_g3, 1), 0.0, 20.0))
    pred_pct = float(round((pred_g3 / 20.0) * 100.0, 1))

    # Pass probability
    if hasattr(clf_pipe.named_steps['model'], "predict_proba"):
        pass_prob = float(clf_pipe.predict_proba(df_input)[0][1]) * 100.0
    else:
        pass_prob = 100.0 if pred_g3 >= 10 else 0.0

    pass_prob = float(round(pass_prob, 1))
    passed_flag = bool(pred_g3 >= 10.0)

    # Risk assessment
    absences = student_dict.get('absences', 0)
    failures = student_dict.get('failures', 0)
    g2 = student_dict.get('G2', 10)
    studytime = student_dict.get('studytime', 2)

    risk_score = 0
    if pred_g3 < 10 or g2 < 10:
        risk_score += 3
    if failures >= 2:
        risk_score += 3
    elif failures == 1:
        risk_score += 1
    if absences > 12:
        risk_score += 2
    elif absences > 6:
        risk_score += 1
    if studytime == 1:
        risk_score += 1

    if risk_score >= 4 or pred_g3 < 10:
        risk_level = "High Risk"
    elif risk_score >= 2 or pred_g3 < 12:
        risk_level = "Moderate Risk"
    else:
        risk_level = "Low Risk"

    letter_grade, letter_color = get_letter_grade(pred_g3)
    recommendations = generate_recommendations(student_dict, pred_g3)

    return {
        'pred_g3': pred_g3,
        'pred_pct': pred_pct,
        'pass_prob': pass_prob,
        'passed': passed_flag,
        'risk_level': risk_level,
        'letter_grade': letter_grade,
        'letter_color': letter_color,
        'recommendations': recommendations
    }

def generate_recommendations(student_dict, pred_g3):
    recs = []
    absences = student_dict.get('absences', 0)
    studytime = student_dict.get('studytime', 2)
    failures = student_dict.get('failures', 0)
    g2 = student_dict.get('G2', 10)
    schoolsup = student_dict.get('schoolsup', 'no')
    higher = student_dict.get('higher', 'yes')
    goout = student_dict.get('goout', 3)
    dalc = student_dict.get('Dalc', 1)
    walc = student_dict.get('Walc', 1)

    if absences > 10:
        recs.append(f"🚨 **High Absences Flagged ({absences} days):** Prioritize classroom attendance. Missing classes is the single fastest compounding risk for falling behind.")
    elif absences > 5:
        recs.append("ℹ️ **Attendance Recommendation:** Aim to keep absences under 4 days to ensure topic continuity.")

    if studytime <= 2:
        curr_label = STUDY_TIME_MAP.get(studytime, "limited")
        recs.append(f"⏱️ **Increase Study Schedule:** Currently logging {curr_label} weekly. Upgrading to 5-10 hours/week consistently yields an average +1.8 point score gain.")
    else:
        recs.append("✅ **Solid Study Dedication:** Strong weekly study habit detected. Maintain deliberate problem-solving exercises.")

    if failures > 0:
        recs.append(f"⚠️ **Targeted Remedial Support:** Student has {failures} prior failed course(s). Pair with a subject tutor to close foundational gaps.")

    if g2 < 10 and schoolsup == 'no':
        recs.append("📚 **Enroll in Academic Support:** Sign up for school-provided tutoring or study lab sessions before the final exam.")

    if (dalc >= 3 or walc >= 4) or (goout >= 4 and studytime <= 2):
        recs.append("⚖️ **Balance Social & Study Time:** Heavy social outings during weekdays appear to conflict with homework and revision time.")

    if higher == 'no':
        recs.append("🎯 **Goal-Setting & Mentorship:** Students with defined higher-education goals score higher. Career guidance could boost motivation.")

    if pred_g3 >= 16:
        recs.append("🌟 **Advanced Opportunity:** Student demonstrates subject mastery. Recommend honors enrichment, competitive exam prep, or peer-tutoring leadership.")

    if not recs:
        recs.append("👍 Student is on a healthy academic trajectory. Continue current study habits and attendance.")

    return recs

def predict_batch_dataset(reg_pipe, clf_pipe, df_in):
    """
    Applies model pipeline on a dataframe and adds rich prediction columns.
    """
    df = df_in.copy()
    for col, default_val in DEFAULT_STUDENT_VALUES.items():
        if col not in df.columns:
            df[col] = default_val

    pred_g3 = reg_pipe.predict(df[FEATURE_COLS])
    pred_g3 = np.clip(np.round(pred_g3, 1), 0.0, 20.0)

    if hasattr(clf_pipe.named_steps['model'], "predict_proba"):
        pass_prob = clf_pipe.predict_proba(df[FEATURE_COLS])[:, 1] * 100.0
    else:
        pass_prob = (pred_g3 >= 10.0).astype(float) * 100.0

    df['Predicted_G3'] = pred_g3
    df['Predicted_Percentage'] = np.round((pred_g3 / 20.0) * 100.0, 1)
    df['Pass_Probability'] = np.round(pass_prob, 1)
    df['Predicted_Status'] = np.where(pred_g3 >= 10.0, 'Pass', 'Fail')
    df['Letter_Grade'] = df['Predicted_G3'].apply(lambda x: get_letter_grade(x)[0].split(' ')[0])

    def row_risk(row):
        score = 0
        if row['Predicted_G3'] < 10 or row.get('G2', 10) < 10:
            score += 3
        if row.get('failures', 0) >= 2:
            score += 3
        elif row.get('failures', 0) == 1:
            score += 1
        if row.get('absences', 0) > 12:
            score += 2
        elif row.get('absences', 0) > 6:
            score += 1
        if row.get('studytime', 2) == 1:
            score += 1

        if score >= 4 or row['Predicted_G3'] < 10:
            return "High Risk"
        elif score >= 2 or row['Predicted_G3'] < 12:
            return "Moderate Risk"
        else:
            return "Low Risk"

    df['Predicted_Risk'] = df.apply(row_risk, axis=1)
    return df
