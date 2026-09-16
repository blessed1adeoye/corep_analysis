"""
Predictive models:
  A) Admission prediction
  B) Malaria risk prediction

Outputs metrics + saved models in ./models/.
"""

import os
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, roc_auc_score, confusion_matrix,
                              classification_report, roc_curve)

MODEL_DIR = "models"
CHART_DIR = "charts"
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(CHART_DIR, exist_ok=True)


def _save(fig, name):
    path = os.path.join(CHART_DIR, name)
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"  💾 {path}")


def _boolify(s):
    return s.astype(str).str.lower().isin(["true", "yes", "1", "positive", "pos", "reactive"])


# ---------- Model A: Admission Prediction ----------
def build_admission_dataset(patient, consultations, nursing):
    p = patient.copy()
    if "Age" not in p.columns and "Date Of Birth" in p.columns:
        p["Date Of Birth"] = pd.to_datetime(p["Date Of Birth"], errors="coerce")
        p["Age"] = ((pd.Timestamp.today() - p["Date Of Birth"]).dt.days / 365.25).round(1)

    p["Target_Admitted"] = _boolify(p["Is Admitted"]).astype(int)
    p = p.dropna(subset=["Target_Admitted"])

    # Aggregate consultations per patient
    if not consultations.empty and "Patient Id" in consultations.columns:
        n_cons = consultations.groupby("Patient Id").size().rename("n_consultations")
        p = p.merge(n_cons, left_on="Id", right_index=True, how="left")

    # Aggregate vitals per patient
    if not nursing.empty and "Patient Id" in nursing.columns:
        agg = nursing.groupby("Patient Id").agg(
            avg_sys=("Blood Pressure Systolic", "mean"),
            avg_dia=("Blood Pressure Diastolic", "mean"),
            avg_temp=("Temperature", "mean"),
            avg_pulse=("Pulse Rate", "mean"),
            avg_spo2=("Oxygen Saturation", "mean"),
        )
        p = p.merge(agg, left_on="Id", right_index=True, how="left")

    p["n_consultations"] = p.get("n_consultations", 0).fillna(0)
    return p


def train_classifier(df, target, numeric_features, categorical_features, model_name):
    X = df[numeric_features + categorical_features]
    y = df[target]

    num_pipe = Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("sc", StandardScaler())
    ])
    cat_pipe = Pipeline([
        ("imp", SimpleImputer(strategy="most_frequent")),
        ("oh", OneHotEncoder(handle_unknown="ignore"))
    ])
    pre = ColumnTransformer([
        ("num", num_pipe, numeric_features),
        ("cat", cat_pipe, categorical_features)
    ])

    models = {
        "LogisticRegression": LogisticRegression(max_iter=1000, class_weight="balanced"),
        "RandomForest": RandomForestClassifier(n_estimators=300, random_state=42,
                                                class_weight="balanced"),
        "GradientBoosting": GradientBoostingClassifier(random_state=42),
    }

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y)

    results = {}
    best_name, best_auc, best_pipe = None, -1, None

    for name, clf in models.items():
        pipe = Pipeline([("pre", pre), ("clf", clf)])
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)
        y_prob = pipe.predict_proba(X_test)[:, 1]

        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred, zero_division=0),
            "recall": recall_score(y_test, y_pred, zero_division=0),
            "f1": f1_score(y_test, y_pred, zero_division=0),
            "roc_auc": roc_auc_score(y_test, y_prob),
        }
        results[name] = metrics
        print(f"  [{model_name}] {name}: AUC={metrics['roc_auc']:.3f}  F1={metrics['f1']:.3f}")

        if metrics["roc_auc"] > best_auc:
            best_auc, best_name, best_pipe = metrics["roc_auc"], name, pipe

    # Save best model
    joblib.dump(best_pipe, os.path.join(MODEL_DIR, f"{model_name}_best.pkl"))
    print(f"  🏆 Best {model_name}: {best_name} (AUC={best_auc:.3f}) → saved")

    # ROC curves
    fig, ax = plt.subplots(figsize=(7, 6))
    for name in models:
        pipe = Pipeline([("pre", pre), ("clf", models[name])]).fit(X_train, y_train)
        y_prob = pipe.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        ax.plot(fpr, tpr, label=f"{name} (AUC={results[name]['roc_auc']:.3f})")
    ax.plot([0, 1], [0, 1], "--", color="grey")
    ax.set_xlabel("FPR"); ax.set_ylabel("TPR")
    ax.set_title(f"ROC Curves – {model_name}")
    ax.legend()
    _save(fig, f"ml_{model_name}_roc.png")

    # Feature importance (from RF)
    rf_pipe = Pipeline([("pre", pre),
                        ("clf", RandomForestClassifier(n_estimators=300, random_state=42))])
    rf_pipe.fit(X_train, y_train)
    try:
        ohe = rf_pipe.named_steps["pre"].named_transformers_["cat"].named_steps["oh"]
        cat_names = ohe.get_feature_names_out(categorical_features)
        feat_names = list(numeric_features) + list(cat_names)
        importances = rf_pipe.named_steps["clf"].feature_importances_
        fi = pd.DataFrame({"feature": feat_names, "importance": importances}) \
               .sort_values("importance", ascending=False).head(15)
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(data=fi, y="feature", x="importance", palette="viridis", ax=ax)
        ax.set_title(f"Top Features – {model_name}")
        plt.tight_layout()
        _save(fig, f"ml_{model_name}_features.png")
    except Exception as e:
        print(f"  ⚠️  feature importance skipped: {e}")

    return results, best_name


def model_admission(patient, consultations, nursing):
    print("\n=== MODEL A: Admission Prediction ===")
    df = build_admission_dataset(patient, consultations, nursing)
    if df["Target_Admitted"].nunique() < 2 or len(df) < 30:
        print("  ⚠️  Not enough data / single class. Skipping.")
        return

    numeric = ["Age", "n_consultations", "avg_sys", "avg_dia", "avg_temp",
               "avg_pulse", "avg_spo2"]
    numeric = [c for c in numeric if c in df.columns]
    categorical = [c for c in ["Gender"] if c in df.columns]

    train_classifier(df, "Target_Admitted", numeric, categorical, "admission")


# ---------- Model B: Malaria Risk ----------
def build_malaria_dataset(lab_tests, patient, nursing):
    if lab_tests.empty or "Malaria Parasite" not in lab_tests.columns:
        return None

    df = lab_tests.copy()
    df["Target_Malaria"] = _boolify(df["Malaria Parasite"]).astype(int)

    df = df.merge(patient[["Id", "Age", "Gender"]],
                  left_on="Patient Id", right_on="Id", how="left")

    if not nursing.empty and "Patient Id" in nursing.columns:
        agg = nursing.groupby("Patient Id").agg(
            avg_temp=("Temperature", "mean"),
            avg_pulse=("Pulse Rate", "mean"),
        )
        df = df.merge(agg, left_on="Patient Id", right_index=True, how="left")

    return df


def model_malaria(lab_tests, patient, nursing):
    print("\n=== MODEL B: Malaria Risk Prediction ===")
    df = build_malaria_dataset(lab_tests, patient, nursing)
    if df is None or df["Target_Malaria"].nunique() < 2 or len(df) < 30:
        print("  ⚠️  Not enough data / single class. Skipping.")
        return
    numeric = [c for c in ["Age", "avg_temp", "avg_pulse"] if c in df.columns]
    categorical = [c for c in ["Gender"] if c in df.columns]
    train_classifier(df, "Target_Malaria", numeric, categorical, "malaria")


# ---------- Entrypoint ----------
if __name__ == "__main__":
    import main_analysis as analysis
    d = analysis.clean_all(analysis.load_data())
    model_admission(d["patient"], d["consultations"], d["nursing"])
    model_malaria(d["lab_tests"], d["patient"], d["nursing"])
    print("\n✅ Models saved in ./models/")