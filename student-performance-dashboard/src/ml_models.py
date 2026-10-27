import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

from src.data_processing import (
    FEATURE_COLS, NUMERICAL_COLS, CATEGORICAL_COLS, get_preprocessor
)

def train_and_evaluate_models(df, test_size=0.2, random_state=42):
    """
    Trains multiple regression and classification models, benchmarks them,
    and returns models, metrics, and feature importances.
    """
    # Ensure features exist
    available_features = [c for c in FEATURE_COLS if c in df.columns]
    X = df[available_features].copy()
    y_reg = df['G3'].values
    y_clf = (df['G3'] >= 10).astype(int).values

    # Train/test split
    X_train, X_test, y_reg_train, y_reg_test, y_clf_train, y_clf_test = train_test_split(
        X, y_reg, y_clf, test_size=test_size, random_state=random_state, stratify=y_clf
    )

    # Preprocessor
    num_cols = [c for c in NUMERICAL_COLS if c in available_features]
    cat_cols = [c for c in CATEGORICAL_COLS if c in available_features]

    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import OneHotEncoder, StandardScaler

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), num_cols),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat_cols)
        ]
    )

    # 1. Regression Models
    reg_candidates = {
        'Ridge Regression': Ridge(alpha=1.0),
        'Random Forest Regressor': RandomForestRegressor(n_estimators=120, random_state=random_state, max_depth=8),
        'Gradient Boosting Regressor': GradientBoostingRegressor(n_estimators=100, random_state=random_state, max_depth=4, learning_rate=0.08)
    }

    reg_results = []
    reg_models = {}
    reg_predictions = {}

    for name, model in reg_candidates.items():
        pipe = Pipeline(steps=[('prep', preprocessor), ('model', model)])
        pipe.fit(X_train, y_reg_train)
        y_pred = pipe.predict(X_test)
        # Clip to realistic exam bounds 0-20
        y_pred_clipped = np.clip(y_pred, 0, 20)

        r2 = r2_score(y_reg_test, y_pred_clipped)
        rmse = float(np.sqrt(mean_squared_error(y_reg_test, y_pred_clipped)))
        mae = float(mean_absolute_error(y_reg_test, y_pred_clipped))

        reg_results.append({
            'Model': name,
            'R² Score': round(r2, 4),
            'RMSE': round(rmse, 3),
            'MAE': round(mae, 3)
        })
        reg_models[name] = pipe
        reg_predictions[name] = {
            'y_true': y_reg_test,
            'y_pred': y_pred_clipped
        }

    # 2. Classification Models (Pass/Fail)
    clf_candidates = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=random_state),
        'Random Forest Classifier': RandomForestClassifier(n_estimators=120, random_state=random_state, max_depth=8),
        'Gradient Boosting Classifier': GradientBoostingClassifier(n_estimators=100, random_state=random_state, max_depth=4, learning_rate=0.08)
    }

    clf_results = []
    clf_models = {}
    clf_predictions = {}

    for name, model in clf_candidates.items():
        pipe = Pipeline(steps=[('prep', preprocessor), ('model', model)])
        pipe.fit(X_train, y_clf_train)
        y_pred = pipe.predict(X_test)
        if hasattr(model, "predict_proba"):
            y_prob = pipe.predict_proba(X_test)[:, 1]
            auc = roc_auc_score(y_clf_test, y_prob)
        else:
            y_prob = y_pred
            auc = 0.5

        acc = accuracy_score(y_clf_test, y_pred)
        prec = precision_score(y_clf_test, y_pred, zero_division=0)
        rec = recall_score(y_clf_test, y_pred, zero_division=0)
        f1 = f1_score(y_clf_test, y_pred, zero_division=0)

        clf_results.append({
            'Model': name,
            'Accuracy': round(acc, 4),
            'Precision': round(prec, 4),
            'Recall': round(rec, 4),
            'F1 Score': round(f1, 4),
            'ROC AUC': round(auc, 4)
        })
        clf_models[name] = pipe
        clf_predictions[name] = {
            'y_true': y_clf_test,
            'y_pred': y_pred,
            'y_prob': y_prob
        }

    # Extract Feature Importances from Best Tree Model (Random Forest Regressor)
    rf_reg_pipe = reg_models['Random Forest Regressor']
    prep = rf_reg_pipe.named_steps['prep']
    rf_model = rf_reg_pipe.named_steps['model']

    cat_feature_names = prep.named_transformers_['cat'].get_feature_names_out(cat_cols).tolist()
    all_feature_names = num_cols + cat_feature_names
    importances = rf_model.feature_importances_

    feat_imp_df = pd.DataFrame({
        'Feature': all_feature_names,
        'Importance': importances
    }).sort_values(by='Importance', ascending=False)

    # Also aggregate one-hot categorical features back to original base features for clearer user charts
    base_importances = {}
    for feat, imp in zip(all_feature_names, importances):
        base_name = feat.split('_')[0] if feat not in num_cols else feat
        base_importances[base_name] = base_importances.get(base_name, 0.0) + imp

    base_feat_imp_df = pd.DataFrame(list(base_importances.items()), columns=['Feature', 'Importance']).sort_values(by='Importance', ascending=False)

    return {
        'reg_results': pd.DataFrame(reg_results).sort_values(by='R² Score', ascending=False),
        'clf_results': pd.DataFrame(clf_results).sort_values(by='Accuracy', ascending=False),
        'reg_models': reg_models,
        'clf_models': clf_models,
        'reg_predictions': reg_predictions,
        'clf_predictions': clf_predictions,
        'feat_imp_df': feat_imp_df,
        'base_feat_imp_df': base_feat_imp_df,
        'X_test': X_test,
        'y_reg_test': y_reg_test,
        'y_clf_test': y_clf_test
    }
