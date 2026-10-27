import os
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

STUDY_TIME_MAP = {
    1: "< 2 hours",
    2: "2 to 5 hours",
    3: "5 to 10 hours",
    4: "> 10 hours"
}

TRAVEL_TIME_MAP = {
    1: "< 15 min",
    2: "15 to 30 min",
    3: "30 min to 1 hr",
    4: "> 1 hr"
}

EDU_MAP = {
    0: "None",
    1: "Primary (4th grade)",
    2: "5th - 9th grade",
    3: "Secondary",
    4: "Higher education"
}

FEATURE_COLS = [
    'school', 'sex', 'age', 'address', 'famsize', 'Pstatus',
    'Medu', 'Fedu', 'Mjob', 'Fjob', 'reason', 'guardian',
    'traveltime', 'studytime', 'failures', 'schoolsup', 'famsup',
    'paid', 'activities', 'nursery', 'higher', 'internet', 'romantic',
    'famrel', 'freetime', 'goout', 'Dalc', 'Walc', 'health',
    'absences', 'G1', 'G2'
]

CATEGORICAL_COLS = [
    'school', 'sex', 'address', 'famsize', 'Pstatus',
    'Mjob', 'Fjob', 'reason', 'guardian', 'schoolsup',
    'famsup', 'paid', 'activities', 'nursery', 'higher',
    'internet', 'romantic'
]

NUMERICAL_COLS = [
    'age', 'Medu', 'Fedu', 'traveltime', 'studytime', 'failures',
    'famrel', 'freetime', 'goout', 'Dalc', 'Walc', 'health',
    'absences', 'G1', 'G2'
]

def load_student_dataset(filepath_or_buffer):
    """
    Loads student dataset supporting semicolon or comma separation.
    Applies feature engineering and enrichment for analytics.
    """
    if isinstance(filepath_or_buffer, str):
        if not os.path.exists(filepath_or_buffer):
            raise FileNotFoundError(f"File not found: {filepath_or_buffer}")
        # Detect delimiter
        with open(filepath_or_buffer, 'r', encoding='utf-8') as f:
            first_line = f.readline()
            sep = ';' if ';' in first_line else ','
        df = pd.read_csv(filepath_or_buffer, sep=sep)
    else:
        # Buffer / uploaded file
        try:
            df = pd.read_csv(filepath_or_buffer, sep=';')
            if len(df.columns) <= 2:
                filepath_or_buffer.seek(0)
                df = pd.read_csv(filepath_or_buffer, sep=',')
        except Exception:
            filepath_or_buffer.seek(0)
            df = pd.read_csv(filepath_or_buffer)

    # Ensure required columns exist
    df.columns = [c.strip().replace('"', '') for c in df.columns]

    # Clean numeric types if read as strings
    for col in ['age', 'Medu', 'Fedu', 'traveltime', 'studytime', 'failures', 
                'famrel', 'freetime', 'goout', 'Dalc', 'Walc', 'health', 
                'absences', 'G1', 'G2', 'G3']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

    # Feature engineering
    if 'G3' in df.columns:
        df['G3_pct'] = (df['G3'] / 20.0) * 100.0
        df['passed'] = (df['G3'] >= 10).astype(int)
        df['grade_bracket'] = df['G3'].apply(categorize_grade)
        df['risk_level'] = df.apply(assess_risk, axis=1)

    if 'Dalc' in df.columns and 'Walc' in df.columns:
        df['alcohol_index'] = np.round((df['Dalc'] * 5 + df['Walc'] * 2) / 7.0, 2)

    if 'studytime' in df.columns:
        df['study_time_label'] = df['studytime'].map(STUDY_TIME_MAP).fillna("Unknown")

    if 'Medu' in df.columns and 'Fedu' in df.columns:
        df['parents_edu_avg'] = (df['Medu'] + df['Fedu']) / 2.0

    return df

def categorize_grade(g3):
    if g3 < 10:
        return "Fail (<10)"
    elif g3 < 12:
        return "Sufficient (10-11)"
    elif g3 < 14:
        return "Satisfactory (12-13)"
    elif g3 < 16:
        return "Good (14-15)"
    else:
        return "Excellent (16-20)"

def assess_risk(row):
    score = 0
    if row.get('G3', 10) < 10 or row.get('G2', 10) < 10:
        score += 3
    if row.get('failures', 0) >= 2:
        score += 3
    elif row.get('failures', 0) == 1:
        score += 1
    if row.get('absences', 0) > 15:
        score += 2
    elif row.get('absences', 0) > 8:
        score += 1
    if row.get('studytime', 2) == 1:
        score += 1

    if score >= 4:
        return "High"
    elif score >= 2:
        return "Moderate"
    else:
        return "Low"

def get_preprocessor():
    """
    Creates a scikit-learn ColumnTransformer for standardizing numericals
    and one-hot encoding categoricals.
    """
    return ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), NUMERICAL_COLS),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), CATEGORICAL_COLS)
        ]
    )
