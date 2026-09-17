"""
Predictive models for COREP data:
  A) Admission prediction      (target: Is Admitted)
  B) Malaria risk prediction   (target: Malaria Parasite)

Outputs:
  - models/admission_best.pkl
  - models/malaria_best.pkl
  - charts/ml_admission_roc.png
  - charts/ml_admission_features.png
  - charts/ml_malaria_roc.png
  - charts/ml_malaria_features.png
"""

import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, classification_report,
)

MODEL_DIR = "models"
CHART_DIR = "charts"
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(CHART_DIR, exist_ok=True)


# ============================================================
# Helpers
# ============================================================
def _save(fig, name):
    path = os.path.join(CHART_DIR, name)
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  💾 {path}")


def _boolify(s):
    """Flexible boolean parser — handles True/False, Yes/No, Positive/Negative, 1/0."""
    return s.astype(str).str.strip().str.lower().isin([
        "true", "yes", "y", "1", "1.0",
        "positive", "pos", "+", "reactive", "detected",
    ])


def _coerce_id(series):
    """Normalize IDs to Int64 so merges work across sheets."""
    return pd.to_numeric(series, errors="coerce").astype("Int64")


def _safe_cols(df, cols):
    """Return only the columns that exist AND have non-null data."""
    return [c for c in cols if c in df.columns and df[c].notna().any()]


# ============================================================
# Dataset builders
# ============================================================
def build_admission_dataset(patient, consultations, nursing):
    """One row per patient with demographics + clinical aggregates + target."""
    if patient is None or patient.empty:
        print("  ⚠️  Empty patient sheet.")
        return pd.DataFrame()

    if "Is Admitted" not in patient.columns:
        print("  ⚠️  No 'Is Admitted' column in Patients sheet.")
        return pd.DataFrame()

    p = patient.copy()

    # Age from DOB if not present
    if "Age" not in p.columns and "Date Of Birth" in p.columns:
        p["Date Of Birth"] = pd.to_datetime(p["Date Of Birth"], errors="coerce")
        p["Age"] = ((pd.Timestamp.today() - p["Date Of Birth"]).dt.days / 365.25).round(1)

    # Normalize IDs for merges
    if "Id" in p.columns:
        p["Id"] = _coerce_id(p["Id"])

    # Target
    p["Target_Admitted"] = _boolify(p["Is Admitted"]).astype(int)

    # Consultation aggregates
    if consultations is not None and not consultations.empty \
            and "Patient Id" in consultations.columns:
        c = consultations.copy()
        c["Patient Id"] = _coerce_id(c["Patient Id"])
        n_cons = c.groupby("Patient Id").size().rename("n_consultations")
        p = p.merge(n_cons, left_on="Id", right_index=True, how="left")

    # Nursing aggregates
    if nursing is not None and not nursing.empty \
            and "Patient Id" in nursing.columns:
        n = nursing.copy()
        n["Patient Id"] = _coerce_id(n["Patient Id"])
        agg_cols = {}
        for src, dst in [
            ("Blood Pressure Systolic", "avg_sys"),
            ("Blood Pressure Diastolic", "avg_dia"),
            ("Temperature", "avg_temp"),
            ("Pulse Rate", "avg_pulse"),
            ("Oxygen Saturation", "avg_spo2"),
            ("Respiratory Rate", "avg_rr"),
        ]:
            if src in n.columns and n[src].notna().any():
                agg_cols[dst] = (src, "mean")
        if agg_cols:
            agg = n.groupby("Patient Id").agg(**agg_cols)
            p = p.merge(agg, left_on="Id", right_index=True, how="left")

    # Fill missing counts
    if "n_consultations" in p.columns:
        p["n_consultations"] = p["n_consultations"].fillna(0)

    return p


def build_malaria_dataset(lab_tests, patient, nursing):
    """One row per lab test with patient demographics + vitals + target."""
    if lab_tests is None or lab_tests.empty:
        print("  ⚠️  Empty lab_tests sheet.")
        return pd.DataFrame()

    if "Malaria Parasite" not in lab_tests.columns:
        print("  ⚠️  No 'Malaria Parasite' column in Lab_Tests sheet.")
        return pd.DataFrame()

    df = lab_tests.copy()
    df["Target_Malaria"] = _boolify(df["Malaria Parasite"]).astype(int)

    # Normalize IDs
    if "Patient Id" in df.columns:
        df["Patient Id"] = _coerce_id(df["Patient Id"])

    # Join patient info
    if patient is not None and not patient.empty:
        p = patient.copy()
        if "Id" in p.columns:
            p["Id"] = _coerce_id(p["Id"])
        if "Age" not in p.columns and "Date Of Birth" in p.columns:
            p["Date Of Birth"] = pd.to_datetime(p["Date Of Birth"], errors="coerce")
            p["Age"] = ((pd.Timestamp.today() - p["Date Of Birth"]).dt.days / 365.25).round(1)

        keep = [c for c in ["Id", "Age", "Gender"] if c in p.columns]
        if keep:
            df = df.merge(p[keep], left_on="Patient Id", right_on="Id", how="left")

    # Join nursing vitals
    if nursing is not None and not nursing.empty \
            and "Patient Id" in nursing.columns:
        n = nursing.copy()
        n["Patient Id"] = _coerce_id(n["Patient Id"])
        agg_cols = {}
        for src, dst in [
            ("Temperature", "avg_temp"),
            ("Pulse Rate", "avg_pulse"),
            ("Blood Pressure Systolic", "avg_sys"),
            ("Blood Pressure Diastolic", "avg_dia"),
        ]:
            if src in n.columns and n[src].notna().any():
                agg_cols[dst] = (src, "mean")
        if agg_cols:
            agg = n.groupby("Patient Id").agg(**agg_cols)
            df = df.merge(agg, left_on="Patient Id", right_index=True, how="left")

    return df


# ============================================================
# Training
# ============================================================
def train_classifier(df, target, numeric_features, categorical_features, model_name):
    """Train 3 models, save the best, plot ROC + feature importance."""

    X = df[numeric_features + categorical_features].copy()
    y = df[target].astype(int).copy()

    # Convert categorical → string to avoid mixed-type issues in OneHotEncoder
    for c in categorical_features:
        X[c] = X[c].astype(str)

    # Preprocessing pipelines
    num_pipe = Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("sc", StandardScaler()),
    ])
    cat_pipe = Pipeline([
        ("imp", SimpleImputer(strategy="most_frequent")),
        ("oh", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    transformers = []
    if numeric_features:
        transformers.append(("num", num_pipe, numeric_features))
    if categorical_features:
        transformers.append(("cat", cat_pipe, categorical_features))

    pre = ColumnTransformer(transformers)

    models = {
        "LogisticRegression": LogisticRegression(max_iter=1000,
                                                  class_weight="balanced"),
        "RandomForest": RandomForestClassifier(n_estimators=300, random_state=42,
                                                class_weight="balanced"),
        "GradientBoosting": GradientBoostingClassifier(random_state=42),
    }

    # Split (stratified)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    results = {}
    best_name, best_auc, best_pipe = None, -1, None

    for name, clf in models.items():
        pipe = Pipeline([("pre", pre), ("clf", clf)])
        try:
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
            print(f"  [{model_name}] {name:20s}  AUC={metrics['roc_auc']:.3f}  "
                  f"F1={metrics['f1']:.3f}  Acc={metrics['accuracy']:.3f}")

            if metrics["roc_auc"] > best_auc:
                best_auc, best_name, best_pipe = metrics["roc_auc"], name, pipe
        except Exception as e:
            print(f"  [{model_name}] {name} failed: {e}")

    if best_pipe is None:
        print(f"  ⚠️  No model trained successfully for {model_name}.")
        return results, None

    # Save best model
    model_path = os.path.join(MODEL_DIR, f"{model_name}_best.pkl")
    joblib.dump(best_pipe, model_path)
    print(f"  🏆 Best {model_name}: {best_name} (AUC={best_auc:.3f}) → {model_path}")

    # ROC curves
    fig, ax = plt.subplots(figsize=(8, 6.5))
    for name in results:
        pipe = Pipeline([("pre", pre), ("clf", models[name])])
        pipe.fit(X_train, y_train)
        y_prob = pipe.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        ax.plot(fpr, tpr, linewidth=2,
                label=f"{name} (AUC={results[name]['roc_auc']:.3f})")
    ax.plot([0, 1], [0, 1], "--", color="grey", linewidth=1)
    ax.set_xlabel("False Positive Rate", fontsize=11)
    ax.set_ylabel("True Positive Rate", fontsize=11)
    ax.set_title(f"ROC Curves – {model_name.title()}", fontsize=13, fontweight="bold")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    _save(fig, f"ml_{model_name}_roc.png")

    # Feature importance (Random Forest)
    try:
        rf_pipe = Pipeline([
            ("pre", pre),
            ("clf", RandomForestClassifier(n_estimators=300, random_state=42)),
        ])
        rf_pipe.fit(X_train, y_train)

        # Extract feature names after preprocessing
        fitted_pre = rf_pipe.named_steps["pre"]
        feat_names = []
        for tname, _, tcols in fitted_pre.transformers_:
            if tname == "num":
                feat_names.extend(tcols)
            elif tname == "cat":
                ohe = fitted_pre.named_transformers_["cat"].named_steps["oh"]
                feat_names.extend(ohe.get_feature_names_out(tcols))
            elif tname == "remainder":
                continue

        importances = rf_pipe.named_steps["clf"].feature_importances_
        if len(feat_names) == len(importances):
            fi = pd.DataFrame({"feature": feat_names, "importance": importances})
            fi = fi.sort_values("importance", ascending=False).head(15)

            fig, ax = plt.subplots(figsize=(11, 7))
            sns.barplot(data=fi, y="feature", x="importance",
                        palette="viridis", ax=ax)
            ax.set_title(f"Top 15 Features – {model_name.title()}",
                         fontsize=13, fontweight="bold")
            ax.set_xlabel("Importance")
            ax.set_ylabel("")
            plt.tight_layout()
            _save(fig, f"ml_{model_name}_features.png")
        else:
            print(f"  ℹ️  Skipping feature plot (name/importance mismatch).")
    except Exception as e:
        print(f"  ⚠️  Feature importance skipped: {e}")

    # Print full classification report for best model
    print(f"\n  Classification report ({best_name}):")
    y_pred = best_pipe.predict(X_test)
    print(classification_report(y_test, y_pred, zero_division=0,
                                 target_names=[f"Not {model_name}", model_name]))

    return results, best_name


# ============================================================
# Model A: Admission
# ============================================================
def model_admission(patient, consultations, nursing):
    print("\n=== MODEL A: Admission Prediction ===")
    df = build_admission_dataset(patient, consultations, nursing)

    if df.empty:
        print("  ⚠️  Empty dataset – skipping.")
        return

    print(f"  📊 Dataset: {len(df)} patients")
    print(f"  📊 Target distribution: {df['Target_Admitted'].value_counts().to_dict()}")

    if df["Target_Admitted"].nunique() < 2:
        uniq = df["Target_Admitted"].unique().tolist()
        print(f"  ⚠️  Only ONE class in target: {uniq}")
        print(f"      → Check the 'Is Admitted' column in the Patients sheet.")
        print(f"      → Accepted positives: true/yes/y/1/positive/pos/+/reactive/detected")
        return

    if len(df) < 10:
        print(f"  ⚠️  Only {len(df)} rows – too few for training (need ≥10).")
        return

    numeric = _safe_cols(df, [
        "Age", "n_consultations",
        "avg_sys", "avg_dia", "avg_temp", "avg_pulse", "avg_spo2", "avg_rr",
    ])
    categorical = [c for c in ["Gender"] if c in df.columns
                   and df[c].notna().any()]

    if not numeric and not categorical:
        print("  ⚠️  No usable features. Skipping.")
        return

    print(f"  🔢 Numeric features ({len(numeric)}): {numeric}")
    print(f"  🏷️  Categorical features ({len(categorical)}): {categorical}")

    train_classifier(df, "Target_Admitted", numeric, categorical, "admission")


# ============================================================
# Model B: Malaria
# ============================================================
def model_malaria(lab_tests, patient, nursing):
    print("\n=== MODEL B: Malaria Risk Prediction ===")
    df = build_malaria_dataset(lab_tests, patient, nursing)

    if df is None or df.empty:
        print("  ⚠️  Empty dataset – skipping.")
        return

    print(f"  📊 Dataset: {len(df)} lab tests")
    print(f"  📊 Target distribution: {df['Target_Malaria'].value_counts().to_dict()}")

    if df["Target_Malaria"].nunique() < 2:
        uniq = df["Target_Malaria"].unique().tolist()
        print(f"  ⚠️  Only ONE class in target: {uniq}")
        print(f"      → Check the 'Malaria Parasite' column in Lab_Tests sheet.")
        print(f"      → Accepted positives: positive/pos/+/reactive/1/true/detected")
        return

    if len(df) < 10:
        print(f"  ⚠️  Only {len(df)} rows – too few for training (need ≥10).")
        return

    numeric = _safe_cols(df, [
        "Age",
        "avg_temp", "avg_pulse", "avg_sys", "avg_dia",
    ])
    categorical = [c for c in ["Gender"] if c in df.columns
                   and df[c].notna().any()]

    if not numeric and not categorical:
        print("  ⚠️  No usable features. Skipping.")
        return

    print(f"  🔢 Numeric features ({len(numeric)}): {numeric}")
    print(f"  🏷️  Categorical features ({len(categorical)}): {categorical}")

    train_classifier(df, "Target_Malaria", numeric, categorical, "malaria")


# ============================================================
# Entrypoint
# ============================================================
if __name__ == "__main__":
    import main_analysis as analysis

    print("\n" + "=" * 60)
    print(" COREP PREDICTIVE MODELS")
    print("=" * 60)

    d = analysis.clean_all(analysis.load_data())

    model_admission(d["patient"], d["consultations"], d["nursing"])
    model_malaria(d["lab_tests"], d["patient"], d["nursing"])

    print("\n✅ Models saved in ./models/")
    print("   - admission_best.pkl")
    print("   - malaria_best.pkl")