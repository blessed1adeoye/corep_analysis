"""
COREP Data Analysis — Complete Pipeline
- Auto-detects sheet names (handles 'patient' vs 'Patients' etc.)
- Cleans data safely
- Generates and SAVES every chart to ./charts/
- Never crashes on missing sheets or columns
- Filters placeholder values (e.g. "Prescription from Treatment Plan") from drug charts
"""

import os
import re
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # non-interactive backend for saving
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore")
sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (12, 6)
plt.rcParams["font.size"] = 10

# ---------------- Configuration ----------------
EXCEL_FILE = "corep_data.xlsx"
CHART_DIR = "charts"
os.makedirs(CHART_DIR, exist_ok=True)

_chart_counter = {"n": 0}


# ============================================================
# Drug-name filtering (removes placeholder values)
# ============================================================
NON_DRUG_VALUES = {
    "prescription from treatment plan",
    "prescription",
    "prescribed",
    "from treatment plan",
    "treatment plan",
    "see treatment plan",
    "as prescribed",
    "as directed",
    "refer to pharmacy",
    "n/a", "na", "none", "nil", "not specified", "unknown",
    "-", "--", "",
}

PLACEHOLDER_PATTERN = re.compile(
    r"^(?:"
    r"prescription(?:\s+from\s+treatment\s+plan)?|"
    r"prescribed|"
    r"from\s+treatment\s+plan|"
    r"treatment\s+plan|"
    r"see\s+(?:the\s+)?treatment\s+plan|"
    r"as\s+(?:prescribed|directed)|"
    r"refer\s+to\s+pharmacy|"
    r"n/?a|none|nil|unknown|not\s+specified|"
    r"[-\s]*"
    r")$",
    re.IGNORECASE,
)


def filter_real_drugs(series):
    """Remove placeholder / non-drug entries from a Drug Name series."""
    if series is None or len(series) == 0:
        return pd.Series(dtype=str)
    s = series.dropna().astype(str).str.strip()
    mask = ~s.str.lower().isin(NON_DRUG_VALUES)
    mask &= ~s.str.match(PLACEHOLDER_PATTERN)
    mask &= s.str.len() > 2
    return s[mask]


# ============================================================
# Utility
# ============================================================
def save_chart(name, fig=None):
    """Save current or given figure and close it."""
    fig = fig or plt.gcf()
    _chart_counter["n"] += 1
    filename = f"{_chart_counter['n']:02d}_{name}.png"
    path = os.path.join(CHART_DIR, filename)
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"  💾 Saved -> {path}")
    return path


def _match_sheet(name, available):
    """Case-insensitive + fuzzy match against available sheet names."""
    name_l = name.lower().strip()
    for s in available:
        if s.lower().strip() == name_l:
            return s
    for s in available:
        if name_l in s.lower().strip() or s.lower().strip() in name_l:
            return s
    for s in available:
        if s.lower().rstrip("s") == name_l.rstrip("s"):
            return s
    return None


def _safe_patient_slice(patient, cols):
    """Return only the requested columns that actually exist."""
    if patient is None or patient.empty:
        return pd.DataFrame(columns=cols)
    return patient[[c for c in cols if c in patient.columns]]


# ============================================================
# Load & Clean
# ============================================================
def load_data(path=EXCEL_FILE):
    """Load all sheets with fuzzy name matching."""
    xls = pd.ExcelFile(path)
    available = xls.sheet_names
    print(f"📑 Sheets in workbook: {available}\n")

    wanted = {
        "patient":       ["patient", "patients"],
        "nursing":       ["Nursing_Assessments", "Nursing_Assessment", "Nursing"],
        "consultations": ["Consultations", "Consultation"],
        "lab_tests":     ["Lab_Tests", "Lab_Test", "Labs"],
        "optical":       ["Optical_Assessments", "Optical_Assessment", "Optical"],
        "pharmacy":      ["Pharmacy_Orders", "Pharmacy_Order", "Pharmacy"],
        "drugs":         ["Drugs", "Drug"],
    }

    data = {}
    for key, candidates in wanted.items():
        matched = None
        for c in candidates:
            matched = _match_sheet(c, available)
            if matched:
                break
        if matched:
            try:
                data[key] = pd.read_excel(path, sheet_name=matched)
                print(f"✅ Loaded {matched:25s} -> key={key:14s} shape={data[key].shape}")
            except Exception as e:
                print(f"⚠️  Failed to read {matched}: {e}")
                data[key] = pd.DataFrame()
        else:
            print(f"❌ No matching sheet for '{key}' (tried {candidates})")
            data[key] = pd.DataFrame()

    return data


def clean(df, date_cols=()):
    """Basic cleaning: strip column names, parse dates, drop fully-empty rows."""
    if df is None or df.empty:
        return pd.DataFrame()
    df = df.copy()
    df.columns = df.columns.astype(str).str.strip()
    for c in date_cols:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors="coerce")
    return df.dropna(how="all")


def clean_all(data):
    """Clean every sheet; derive Age & Age Group for patients."""
    date_map = {
        "patient":       ["Created At", "Updated At", "Date Of Birth"],
        "nursing":       ["Created At", "Updated At", "Completed At"],
        "consultations": ["Created At", "Updated At", "Completed At"],
        "lab_tests":     ["Created At", "Updated At", "Completed At", "Viewed At"],
        "optical":       ["Created At", "Updated At", "Completed At",
                          "Viewed At", "Viewed By Physician At"],
        "pharmacy":      ["Created At", "Updated At", "Dispensed At"],
        "drugs":         ["Created At", "Updated At"],
    }

    for k, df in data.items():
        if df is None or df.empty:
            continue
        data[k] = clean(df, date_map.get(k, []))

    p = data.get("patient")
    if p is not None and not p.empty and "Date Of Birth" in p.columns:
        today = pd.Timestamp.today()
        p["Age"] = ((today - p["Date Of Birth"]).dt.days / 365.25).round(1)
        bins   = [0, 5, 12, 18, 35, 50, 65, 120]
        labels = ["0-5", "6-12", "13-18", "19-35", "36-50", "51-65", "65+"]
        p["Age Group"] = pd.cut(p["Age"], bins=bins, labels=labels)
        data["patient"] = p

    return data


# ============================================================
# Charts
# ============================================================
def chart_patient_demographics(patient):
    if patient is None or patient.empty:
        print("  ⚠️  No patient data – skipping demographics.")
        return

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    if "Gender" in patient.columns and patient["Gender"].notna().any():
        patient["Gender"].value_counts().plot(kind="bar", ax=axes[0, 0], color="steelblue")
        axes[0, 0].set_title("Gender Distribution")
    else:
        axes[0, 0].axis("off")

    if "Age" in patient.columns and patient["Age"].notna().any():
        sns.histplot(patient["Age"].dropna(), bins=20, kde=True, ax=axes[0, 1], color="teal")
        axes[0, 1].set_title("Age Distribution")
    else:
        axes[0, 1].axis("off")

    if "Age Group" in patient.columns and patient["Age Group"].notna().any():
        patient["Age Group"].value_counts().sort_index().plot(
            kind="bar", ax=axes[1, 0], color="coral")
        axes[1, 0].set_title("Age Group Distribution")
    else:
        axes[1, 0].axis("off")

    if "Is Admitted" in patient.columns and patient["Is Admitted"].notna().any():
        patient["Is Admitted"].value_counts().plot(
            kind="pie", autopct="%1.1f%%", ax=axes[1, 1],
            colors=["#66b3ff", "#ff9999"])
        axes[1, 1].set_title("Admission Status")
        axes[1, 1].set_ylabel("")
    else:
        axes[1, 1].axis("off")

    plt.tight_layout()
    save_chart("patient_demographics", fig)


def chart_registration_trends(patient):
    if patient is None or patient.empty or "Created At" not in patient.columns:
        print("  ⚠️  No patient registration dates – skipping trend.")
        return
    if patient["Created At"].isna().all():
        return

    p = patient.copy()
    p["Month"] = p["Created At"].dt.to_period("M").astype(str)
    monthly = p.groupby("Month").size()

    fig, ax = plt.subplots(figsize=(14, 5))
    monthly.plot(kind="line", marker="o", color="navy", ax=ax)
    ax.set_title("Monthly Patient Registrations")
    ax.set_ylabel("New Patients")
    plt.tight_layout()
    save_chart("monthly_registrations", fig)

    p["DayOfWeek"] = p["Created At"].dt.day_name()
    order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.countplot(data=p, x="DayOfWeek", order=order, palette="viridis", ax=ax)
    ax.set_title("Registrations by Day of Week")
    plt.xticks(rotation=45)
    plt.tight_layout()
    save_chart("registrations_by_dayofweek", fig)


def chart_vitals(nursing):
    if nursing is None or nursing.empty:
        print("  ⚠️  No nursing data – skipping vitals.")
        return

    vitals = ["Blood Pressure Systolic", "Blood Pressure Diastolic",
              "Pulse Rate", "Temperature", "Respiratory Rate", "Oxygen Saturation"]
    vitals = [c for c in vitals if c in nursing.columns]
    if not vitals:
        print("  ⚠️  No vitals columns found.")
        return

    fig, axes = plt.subplots(2, 3, figsize=(16, 8))
    for ax, col in zip(axes.flatten(), vitals):
        sns.boxplot(y=nursing[col].dropna(), ax=ax, color="lightblue")
        ax.set_title(col)
    for ax in axes.flatten()[len(vitals):]:
        ax.axis("off")
    plt.tight_layout()
    save_chart("vitals_boxplots", fig)

    fig, axes = plt.subplots(2, 3, figsize=(16, 8))
    for ax, col in zip(axes.flatten(), vitals):
        sns.histplot(nursing[col].dropna(), kde=True, ax=ax, color="seagreen")
        ax.set_title(f"Distribution – {col}")
    for ax in axes.flatten()[len(vitals):]:
        ax.axis("off")
    plt.tight_layout()
    save_chart("vitals_distributions", fig)

    fig, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(nursing[vitals].corr(), annot=True, cmap="coolwarm", fmt=".2f", ax=ax)
    ax.set_title("Vitals Correlation Matrix")
    plt.tight_layout()
    save_chart("vitals_correlation", fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    if "Biohazard Risk" in nursing.columns:
        nursing["Biohazard Risk"].value_counts().plot(kind="bar", ax=axes[0], color="tomato")
        axes[0].set_title("Biohazard Risk")
    else:
        axes[0].axis("off")
    if "Isolation Required" in nursing.columns:
        nursing["Isolation Required"].value_counts().plot(kind="bar", ax=axes[1], color="orange")
        axes[1].set_title("Isolation Required")
    else:
        axes[1].axis("off")
    plt.tight_layout()
    save_chart("biohazard_isolation", fig)


def chart_consultations(consultations, patient):
    if consultations is None or consultations.empty:
        print("  ⚠️  No consultations data – skipping.")
        return
    if "Diagnosis" not in consultations.columns:
        print("  ⚠️  No 'Diagnosis' column – skipping.")
        return

    top_diag = consultations["Diagnosis"].value_counts().head(15)
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.barplot(y=top_diag.index, x=top_diag.values, palette="magma", ax=ax)
    ax.set_title("Top 15 Diagnoses")
    plt.tight_layout()
    save_chart("top15_diagnoses", fig)

    pat_slice = _safe_patient_slice(patient, ["Id", "Age", "Gender", "Age Group"])
    if (not pat_slice.empty
            and "Patient Id" in consultations.columns
            and "Id" in pat_slice.columns):
        cons_p = consultations.merge(
            pat_slice, left_on="Patient Id", right_on="Id",
            suffixes=("", "_pat"), how="left")
    else:
        cons_p = consultations.copy()
        print("  ℹ️  Patient info unavailable – gender/age breakdown skipped.")

    if "Gender" in cons_p.columns and cons_p["Gender"].notna().any():
        top10 = top_diag.head(10).index
        sub = cons_p[cons_p["Diagnosis"].isin(top10)]
        ct = pd.crosstab(sub["Diagnosis"], sub["Gender"])
        if not ct.empty:
            fig, ax = plt.subplots(figsize=(12, 7))
            ct.plot(kind="barh", stacked=True, colormap="Set2", ax=ax)
            ax.set_title("Top 10 Diagnoses by Gender")
            plt.tight_layout()
            save_chart("diagnoses_by_gender", fig)

    referrals = ["Refer To Pharmacy", "Refer To Laboratory",
                 "Refer To Optician", "Refer To Specialist"]
    referrals = [c for c in referrals if c in consultations.columns]
    if referrals:
        def positive_count(x):
            if x.dtype == bool:
                return int(x.sum())
            return int(x.astype(str).str.lower().isin(["true", "yes", "1"]).sum())
        ref_counts = consultations[referrals].apply(positive_count)
        fig, ax = plt.subplots(figsize=(9, 5))
        ref_counts.plot(kind="bar", color="mediumseagreen", ax=ax)
        ax.set_title("Referral Distribution")
        plt.xticks(rotation=30)
        plt.tight_layout()
        save_chart("referrals", fig)

    if "Created At" in consultations.columns and consultations["Created At"].notna().any():
        cons = consultations.copy()
        cons["Month"] = cons["Created At"].dt.to_period("M").astype(str)
        fig, ax = plt.subplots(figsize=(14, 5))
        cons.groupby("Month").size().plot(kind="line", marker="o", color="purple", ax=ax)
        ax.set_title("Monthly Consultations")
        plt.tight_layout()
        save_chart("monthly_consultations", fig)

        cons["Hour"] = cons["Created At"].dt.hour
        fig, ax = plt.subplots(figsize=(12, 4))
        sns.countplot(data=cons, x="Hour", palette="crest", ax=ax)
        ax.set_title("Consultations by Hour of Day")
        plt.tight_layout()
        save_chart("consultations_by_hour", fig)


def chart_lab_tests(lab_tests):
    if lab_tests is None or lab_tests.empty:
        print("  ⚠️  No lab test data – skipping.")
        return

    lab_cols = [c for c in ["Malaria Parasite", "Random Blood Sugar", "Hbsag"]
                if c in lab_tests.columns]
    if lab_cols:
        fig, axes = plt.subplots(1, len(lab_cols), figsize=(6 * len(lab_cols), 4))
        if len(lab_cols) == 1:
            axes = [axes]
        for ax, col in zip(axes, lab_cols):
            lab_tests[col].value_counts(dropna=False).plot(
                kind="bar", ax=ax, color="indianred")
            ax.set_title(col)
            ax.tick_params(axis="x", rotation=30)
        plt.tight_layout()
        save_chart("lab_test_results", fig)

    if {"Created At", "Completed At"}.issubset(lab_tests.columns):
        tat = (lab_tests["Completed At"] - lab_tests["Created At"]).dt.total_seconds() / 3600
        tat = tat.dropna()
        if not tat.empty:
            fig, ax = plt.subplots(figsize=(10, 5))
            sns.histplot(tat, bins=30, kde=True, color="steelblue", ax=ax)
            ax.set_title("Lab Test Turnaround Time (hours)")
            plt.tight_layout()
            save_chart("lab_turnaround_time", fig)


def chart_optical(optical):
    if optical is None or optical.empty:
        print("  ⚠️  No optical data – skipping.")
        return

    if "Is Walk In" in optical.columns:
        fig, ax = plt.subplots(figsize=(6, 6))
        optical["Is Walk In"].value_counts().plot(
            kind="pie", autopct="%1.1f%%", colors=["#88c999", "#f4a582"], ax=ax)
        ax.set_title("Walk-in vs Referred Optical Patients")
        ax.set_ylabel("")
        plt.tight_layout()
        save_chart("optical_walkin_vs_referred", fig)

    if "Glasses Allocated" in optical.columns:
        fig, ax = plt.subplots(figsize=(8, 5))
        optical["Glasses Allocated"].value_counts().plot(
            kind="bar", color="mediumpurple", ax=ax)
        ax.set_title("Glasses Allocated")
        plt.tight_layout()
        save_chart("glasses_allocated", fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    if "Visual Acuity Left" in optical.columns:
        sns.countplot(data=optical, x="Visual Acuity Left", palette="Blues", ax=axes[0])
        axes[0].set_title("Visual Acuity – Left")
        axes[0].tick_params(axis="x", rotation=45)
    else:
        axes[0].axis("off")
    if "Visual Acuity Right" in optical.columns:
        sns.countplot(data=optical, x="Visual Acuity Right", palette="Greens", ax=axes[1])
        axes[1].set_title("Visual Acuity – Right")
        axes[1].tick_params(axis="x", rotation=45)
    else:
        axes[1].axis("off")
    plt.tight_layout()
    save_chart("visual_acuity", fig)


def chart_pharmacy(pharmacy, drugs):
    # --- Top dispensed drugs (placeholder values filtered out) ---
    if pharmacy is not None and not pharmacy.empty and "Drug Name" in pharmacy.columns:
        clean_names = filter_real_drugs(pharmacy["Drug Name"])
        top_drugs = clean_names.value_counts().head(15)

        if top_drugs.empty:
            print("  ⚠️  No valid drug names after filtering placeholders.")
        else:
            fig, ax = plt.subplots(figsize=(12, 6))
            sns.barplot(y=top_drugs.index, x=top_drugs.values,
                        palette="YlOrRd", ax=ax)
            ax.set_title("Top 15 Dispensed Drugs")
            ax.set_xlabel("Number of Dispensations")
            plt.tight_layout()
            save_chart("top15_drugs", fig)

    # --- Dispense rate ---
    if (pharmacy is not None and not pharmacy.empty
            and "Dispensed" in pharmacy.columns):
        fig, ax = plt.subplots(figsize=(6, 6))
        pharmacy["Dispensed"].value_counts().plot(kind="pie", autopct="%1.1f%%", ax=ax)
        ax.set_title("Pharmacy Dispense Rate")
        ax.set_ylabel("")
        plt.tight_layout()
        save_chart("dispense_rate", fig)

    # --- Stock by category ---
    if (drugs is not None and not drugs.empty
            and {"Quantity", "Reorder Level", "Category"}.issubset(drugs.columns)):
        d = drugs.copy()
        d["Stock Status"] = np.where(d["Quantity"] <= d["Reorder Level"], "Reorder", "OK")
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.barplot(data=d, x="Category", y="Quantity", hue="Stock Status", ax=ax)
        ax.set_title("Drug Stock by Category")
        plt.xticks(rotation=45)
        plt.tight_layout()
        save_chart("stock_by_category", fig)


def chart_patient_journey(patient, consultations, lab_tests, pharmacy, optical):
    def _n(df, col="Patient Id", fallback="Id"):
        if df is None or df.empty:
            return 0
        if col in df.columns:
            return df[col].nunique()
        if fallback in df.columns:
            return df[fallback].nunique()
        return 0

    stages = ["Registered", "Consulted", "Lab Tested", "Pharmacy", "Optical"]
    counts = [
        _n(patient, "Id", "Id"),
        _n(consultations),
        _n(lab_tests),
        _n(pharmacy),
        _n(optical),
    ]
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(x=stages, y=counts, palette="viridis", ax=ax)
    ax.set_title("Patient Care Pathway Funnel")
    ax.set_ylabel("Unique Patients")
    for i, v in enumerate(counts):
        ax.text(i, v + 0.5, str(v), ha="center")
    plt.tight_layout()
    save_chart("patient_journey_funnel", fig)


# ============================================================
# Entry Point
# ============================================================
def run():
    print("\n=== COREP ANALYSIS STARTED ===")
    data = clean_all(load_data())

    print("\n📊 Data availability:")
    for k, df in data.items():
        status = "✅" if (df is not None and not df.empty) else "❌ empty"
        n = 0 if df is None else len(df)
        print(f"   {k:15s} {status}  ({n} rows)")

    print("\n📊 Generating charts ...")
    jobs = [
        ("patient_demographics", lambda: chart_patient_demographics(data["patient"])),
        ("registration_trends",  lambda: chart_registration_trends(data["patient"])),
        ("vitals",               lambda: chart_vitals(data["nursing"])),
        ("consultations",        lambda: chart_consultations(data["consultations"], data["patient"])),
        ("lab_tests",            lambda: chart_lab_tests(data["lab_tests"])),
        ("optical",              lambda: chart_optical(data["optical"])),
        ("pharmacy",             lambda: chart_pharmacy(data["pharmacy"], data["drugs"])),
        ("patient_journey",      lambda: chart_patient_journey(
                                        data["patient"], data["consultations"],
                                        data["lab_tests"], data["pharmacy"], data["optical"])),
    ]
    for name, fn in jobs:
        try:
            fn()
        except Exception as e:
            print(f"  ⚠️  {name} failed: {e}")

    print(f"\n✅ All charts saved to ./{CHART_DIR}/  ({_chart_counter['n']} files)")
    return data


if __name__ == "__main__":
    run()
    
# """
# COREP Data Analysis — Complete Pipeline
# - Auto-detects sheet names (handles 'patient' vs 'Patients' etc.)
# - Cleans data safely
# - Generates and SAVES every chart to ./charts/
# - Never crashes on missing sheets or columns
# """

# import os
# import warnings
# from datetime import datetime

# import numpy as np
# import pandas as pd
# import matplotlib
# matplotlib.use("Agg")  # non-interactive backend for saving
# import matplotlib.pyplot as plt
# import seaborn as sns

# warnings.filterwarnings("ignore")
# sns.set_style("whitegrid")
# plt.rcParams["figure.figsize"] = (12, 6)
# plt.rcParams["font.size"] = 10

# # ---------------- Configuration ----------------
# EXCEL_FILE = "corep_data.xlsx"
# CHART_DIR = "charts"
# os.makedirs(CHART_DIR, exist_ok=True)

# _chart_counter = {"n": 0}


# # ============================================================
# # Utility
# # ============================================================
# def save_chart(name, fig=None):
#     """Save current or given figure and close it."""
#     fig = fig or plt.gcf()
#     _chart_counter["n"] += 1
#     filename = f"{_chart_counter['n']:02d}_{name}.png"
#     path = os.path.join(CHART_DIR, filename)
#     fig.savefig(path, dpi=110, bbox_inches="tight")
#     plt.close(fig)
#     print(f"  💾 Saved -> {path}")
#     return path


# def _match_sheet(name, available):
#     """Case-insensitive + fuzzy match against available sheet names."""
#     name_l = name.lower().strip()
#     # exact (case-insensitive)
#     for s in available:
#         if s.lower().strip() == name_l:
#             return s
#     # startswith / contains
#     for s in available:
#         if name_l in s.lower().strip() or s.lower().strip() in name_l:
#             return s
#     # singular / plural fallback
#     for s in available:
#         if s.lower().rstrip("s") == name_l.rstrip("s"):
#             return s
#     return None


# def _safe_patient_slice(patient, cols):
#     """Return only the requested columns that actually exist."""
#     if patient is None or patient.empty:
#         return pd.DataFrame(columns=cols)
#     return patient[[c for c in cols if c in patient.columns]]


# # ============================================================
# # Load & Clean
# # ============================================================
# def load_data(path=EXCEL_FILE):
#     """Load all sheets with fuzzy name matching."""
#     xls = pd.ExcelFile(path)
#     available = xls.sheet_names
#     print(f"📑 Sheets in workbook: {available}\n")

#     wanted = {
#         "patient":       ["patient", "patients"],
#         "nursing":       ["Nursing_Assessments", "Nursing_Assessment", "Nursing"],
#         "consultations": ["Consultations", "Consultation"],
#         "lab_tests":     ["Lab_Tests", "Lab_Test", "Labs"],
#         "optical":       ["Optical_Assessments", "Optical_Assessment", "Optical"],
#         "pharmacy":      ["Pharmacy_Orders", "Pharmacy_Order", "Pharmacy"],
#         "drugs":         ["Drugs", "Drug"],
#     }

#     data = {}
#     for key, candidates in wanted.items():
#         matched = None
#         for c in candidates:
#             matched = _match_sheet(c, available)
#             if matched:
#                 break
#         if matched:
#             try:
#                 data[key] = pd.read_excel(path, sheet_name=matched)
#                 print(f"✅ Loaded {matched:25s} -> key={key:14s} shape={data[key].shape}")
#             except Exception as e:
#                 print(f"⚠️  Failed to read {matched}: {e}")
#                 data[key] = pd.DataFrame()
#         else:
#             print(f"❌ No matching sheet for '{key}' (tried {candidates})")
#             data[key] = pd.DataFrame()

#     return data


# def clean(df, date_cols=()):
#     """Basic cleaning: strip column names, parse dates, drop fully-empty rows."""
#     if df is None or df.empty:
#         return pd.DataFrame()
#     df = df.copy()
#     df.columns = df.columns.astype(str).str.strip()
#     for c in date_cols:
#         if c in df.columns:
#             df[c] = pd.to_datetime(df[c], errors="coerce")
#     return df.dropna(how="all")


# def clean_all(data):
#     """Clean every sheet; derive Age & Age Group for patients."""
#     date_map = {
#         "patient":       ["Created At", "Updated At", "Date Of Birth"],
#         "nursing":       ["Created At", "Updated At", "Completed At"],
#         "consultations": ["Created At", "Updated At", "Completed At"],
#         "lab_tests":     ["Created At", "Updated At", "Completed At", "Viewed At"],
#         "optical":       ["Created At", "Updated At", "Completed At",
#                           "Viewed At", "Viewed By Physician At"],
#         "pharmacy":      ["Created At", "Updated At", "Dispensed At"],
#         "drugs":         ["Created At", "Updated At"],
#     }

#     for k, df in data.items():
#         if df is None or df.empty:
#             continue
#         data[k] = clean(df, date_map.get(k, []))

#     # Derived age (only if patient sheet exists & has DOB)
#     p = data.get("patient")
#     if p is not None and not p.empty and "Date Of Birth" in p.columns:
#         today = pd.Timestamp.today()
#         p["Age"] = ((today - p["Date Of Birth"]).dt.days / 365.25).round(1)
#         bins   = [0, 5, 12, 18, 35, 50, 65, 120]
#         labels = ["0-5", "6-12", "13-18", "19-35", "36-50", "51-65", "65+"]
#         p["Age Group"] = pd.cut(p["Age"], bins=bins, labels=labels)
#         data["patient"] = p

#     return data


# # ============================================================
# # Charts
# # ============================================================
# def chart_patient_demographics(patient):
#     if patient is None or patient.empty:
#         print("  ⚠️  No patient data – skipping demographics.")
#         return

#     fig, axes = plt.subplots(2, 2, figsize=(14, 10))

#     if "Gender" in patient.columns and patient["Gender"].notna().any():
#         patient["Gender"].value_counts().plot(kind="bar", ax=axes[0, 0], color="steelblue")
#         axes[0, 0].set_title("Gender Distribution")
#     else:
#         axes[0, 0].axis("off")

#     if "Age" in patient.columns and patient["Age"].notna().any():
#         sns.histplot(patient["Age"].dropna(), bins=20, kde=True, ax=axes[0, 1], color="teal")
#         axes[0, 1].set_title("Age Distribution")
#     else:
#         axes[0, 1].axis("off")

#     if "Age Group" in patient.columns and patient["Age Group"].notna().any():
#         patient["Age Group"].value_counts().sort_index().plot(
#             kind="bar", ax=axes[1, 0], color="coral")
#         axes[1, 0].set_title("Age Group Distribution")
#     else:
#         axes[1, 0].axis("off")

#     if "Is Admitted" in patient.columns and patient["Is Admitted"].notna().any():
#         patient["Is Admitted"].value_counts().plot(
#             kind="pie", autopct="%1.1f%%", ax=axes[1, 1],
#             colors=["#66b3ff", "#ff9999"])
#         axes[1, 1].set_title("Admission Status")
#         axes[1, 1].set_ylabel("")
#     else:
#         axes[1, 1].axis("off")

#     plt.tight_layout()
#     save_chart("patient_demographics", fig)


# def chart_registration_trends(patient):
#     if patient is None or patient.empty or "Created At" not in patient.columns:
#         print("  ⚠️  No patient registration dates – skipping trend.")
#         return
#     if patient["Created At"].isna().all():
#         return

#     p = patient.copy()
#     p["Month"] = p["Created At"].dt.to_period("M").astype(str)
#     monthly = p.groupby("Month").size()

#     fig, ax = plt.subplots(figsize=(14, 5))
#     monthly.plot(kind="line", marker="o", color="navy", ax=ax)
#     ax.set_title("Monthly Patient Registrations")
#     ax.set_ylabel("New Patients")
#     plt.tight_layout()
#     save_chart("monthly_registrations", fig)

#     p["DayOfWeek"] = p["Created At"].dt.day_name()
#     order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
#     fig, ax = plt.subplots(figsize=(10, 5))
#     sns.countplot(data=p, x="DayOfWeek", order=order, palette="viridis", ax=ax)
#     ax.set_title("Registrations by Day of Week")
#     plt.xticks(rotation=45)
#     plt.tight_layout()
#     save_chart("registrations_by_dayofweek", fig)


# def chart_vitals(nursing):
#     if nursing is None or nursing.empty:
#         print("  ⚠️  No nursing data – skipping vitals.")
#         return

#     vitals = ["Blood Pressure Systolic", "Blood Pressure Diastolic",
#               "Pulse Rate", "Temperature", "Respiratory Rate", "Oxygen Saturation"]
#     vitals = [c for c in vitals if c in nursing.columns]
#     if not vitals:
#         print("  ⚠️  No vitals columns found.")
#         return

#     # Box plots
#     fig, axes = plt.subplots(2, 3, figsize=(16, 8))
#     for ax, col in zip(axes.flatten(), vitals):
#         sns.boxplot(y=nursing[col].dropna(), ax=ax, color="lightblue")
#         ax.set_title(col)
#     for ax in axes.flatten()[len(vitals):]:
#         ax.axis("off")
#     plt.tight_layout()
#     save_chart("vitals_boxplots", fig)

#     # Distributions
#     fig, axes = plt.subplots(2, 3, figsize=(16, 8))
#     for ax, col in zip(axes.flatten(), vitals):
#         sns.histplot(nursing[col].dropna(), kde=True, ax=ax, color="seagreen")
#         ax.set_title(f"Distribution – {col}")
#     for ax in axes.flatten()[len(vitals):]:
#         ax.axis("off")
#     plt.tight_layout()
#     save_chart("vitals_distributions", fig)

#     # Correlation heatmap
#     fig, ax = plt.subplots(figsize=(9, 7))
#     sns.heatmap(nursing[vitals].corr(), annot=True, cmap="coolwarm", fmt=".2f", ax=ax)
#     ax.set_title("Vitals Correlation Matrix")
#     plt.tight_layout()
#     save_chart("vitals_correlation", fig)

#     # Biohazard / isolation
#     fig, axes = plt.subplots(1, 2, figsize=(12, 4))
#     if "Biohazard Risk" in nursing.columns:
#         nursing["Biohazard Risk"].value_counts().plot(kind="bar", ax=axes[0], color="tomato")
#         axes[0].set_title("Biohazard Risk")
#     else:
#         axes[0].axis("off")
#     if "Isolation Required" in nursing.columns:
#         nursing["Isolation Required"].value_counts().plot(kind="bar", ax=axes[1], color="orange")
#         axes[1].set_title("Isolation Required")
#     else:
#         axes[1].axis("off")
#     plt.tight_layout()
#     save_chart("biohazard_isolation", fig)


# def chart_consultations(consultations, patient):
#     if consultations is None or consultations.empty:
#         print("  ⚠️  No consultations data – skipping.")
#         return
#     if "Diagnosis" not in consultations.columns:
#         print("  ⚠️  No 'Diagnosis' column – skipping.")
#         return

#     # Top 15 diagnoses
#     top_diag = consultations["Diagnosis"].value_counts().head(15)
#     fig, ax = plt.subplots(figsize=(12, 6))
#     sns.barplot(y=top_diag.index, x=top_diag.values, palette="magma", ax=ax)
#     ax.set_title("Top 15 Diagnoses")
#     plt.tight_layout()
#     save_chart("top15_diagnoses", fig)

#     # Safe merge with patient
#     pat_slice = _safe_patient_slice(patient, ["Id", "Age", "Gender", "Age Group"])
#     if (not pat_slice.empty
#             and "Patient Id" in consultations.columns
#             and "Id" in pat_slice.columns):
#         cons_p = consultations.merge(
#             pat_slice, left_on="Patient Id", right_on="Id",
#             suffixes=("", "_pat"), how="left")
#     else:
#         cons_p = consultations.copy()
#         print("  ℹ️  Patient info unavailable – gender/age breakdown skipped.")

#     # Diagnoses by gender
#     if "Gender" in cons_p.columns and cons_p["Gender"].notna().any():
#         top10 = top_diag.head(10).index
#         sub = cons_p[cons_p["Diagnosis"].isin(top10)]
#         ct = pd.crosstab(sub["Diagnosis"], sub["Gender"])
#         if not ct.empty:
#             fig, ax = plt.subplots(figsize=(12, 7))
#             ct.plot(kind="barh", stacked=True, colormap="Set2", ax=ax)
#             ax.set_title("Top 10 Diagnoses by Gender")
#             plt.tight_layout()
#             save_chart("diagnoses_by_gender", fig)

#     # Referral distribution
#     referrals = ["Refer To Pharmacy", "Refer To Laboratory",
#                  "Refer To Optician", "Refer To Specialist"]
#     referrals = [c for c in referrals if c in consultations.columns]
#     if referrals:
#         def positive_count(x):
#             if x.dtype == bool:
#                 return int(x.sum())
#             return int(x.astype(str).str.lower().isin(["true", "yes", "1"]).sum())
#         ref_counts = consultations[referrals].apply(positive_count)
#         fig, ax = plt.subplots(figsize=(9, 5))
#         ref_counts.plot(kind="bar", color="mediumseagreen", ax=ax)
#         ax.set_title("Referral Distribution")
#         plt.xticks(rotation=30)
#         plt.tight_layout()
#         save_chart("referrals", fig)

#     # Monthly consultations + hour-of-day
#     if "Created At" in consultations.columns and consultations["Created At"].notna().any():
#         cons = consultations.copy()
#         cons["Month"] = cons["Created At"].dt.to_period("M").astype(str)
#         fig, ax = plt.subplots(figsize=(14, 5))
#         cons.groupby("Month").size().plot(kind="line", marker="o", color="purple", ax=ax)
#         ax.set_title("Monthly Consultations")
#         plt.tight_layout()
#         save_chart("monthly_consultations", fig)

#         cons["Hour"] = cons["Created At"].dt.hour
#         fig, ax = plt.subplots(figsize=(12, 4))
#         sns.countplot(data=cons, x="Hour", palette="crest", ax=ax)
#         ax.set_title("Consultations by Hour of Day")
#         plt.tight_layout()
#         save_chart("consultations_by_hour", fig)


# def chart_lab_tests(lab_tests):
#     if lab_tests is None or lab_tests.empty:
#         print("  ⚠️  No lab test data – skipping.")
#         return

#     lab_cols = [c for c in ["Malaria Parasite", "Random Blood Sugar", "Hbsag"]
#                 if c in lab_tests.columns]
#     if lab_cols:
#         fig, axes = plt.subplots(1, len(lab_cols), figsize=(6 * len(lab_cols), 4))
#         if len(lab_cols) == 1:
#             axes = [axes]
#         for ax, col in zip(axes, lab_cols):
#             lab_tests[col].value_counts(dropna=False).plot(
#                 kind="bar", ax=ax, color="indianred")
#             ax.set_title(col)
#             ax.tick_params(axis="x", rotation=30)
#         plt.tight_layout()
#         save_chart("lab_test_results", fig)

#     # Turnaround time
#     if {"Created At", "Completed At"}.issubset(lab_tests.columns):
#         tat = (lab_tests["Completed At"] - lab_tests["Created At"]).dt.total_seconds() / 3600
#         tat = tat.dropna()
#         if not tat.empty:
#             fig, ax = plt.subplots(figsize=(10, 5))
#             sns.histplot(tat, bins=30, kde=True, color="steelblue", ax=ax)
#             ax.set_title("Lab Test Turnaround Time (hours)")
#             plt.tight_layout()
#             save_chart("lab_turnaround_time", fig)


# def chart_optical(optical):
#     if optical is None or optical.empty:
#         print("  ⚠️  No optical data – skipping.")
#         return

#     if "Is Walk In" in optical.columns:
#         fig, ax = plt.subplots(figsize=(6, 6))
#         optical["Is Walk In"].value_counts().plot(
#             kind="pie", autopct="%1.1f%%", colors=["#88c999", "#f4a582"], ax=ax)
#         ax.set_title("Walk-in vs Referred Optical Patients")
#         ax.set_ylabel("")
#         plt.tight_layout()
#         save_chart("optical_walkin_vs_referred", fig)

#     if "Glasses Allocated" in optical.columns:
#         fig, ax = plt.subplots(figsize=(8, 5))
#         optical["Glasses Allocated"].value_counts().plot(
#             kind="bar", color="mediumpurple", ax=ax)
#         ax.set_title("Glasses Allocated")
#         plt.tight_layout()
#         save_chart("glasses_allocated", fig)

#     fig, axes = plt.subplots(1, 2, figsize=(12, 4))
#     if "Visual Acuity Left" in optical.columns:
#         sns.countplot(data=optical, x="Visual Acuity Left", palette="Blues", ax=axes[0])
#         axes[0].set_title("Visual Acuity – Left")
#         axes[0].tick_params(axis="x", rotation=45)
#     else:
#         axes[0].axis("off")
#     if "Visual Acuity Right" in optical.columns:
#         sns.countplot(data=optical, x="Visual Acuity Right", palette="Greens", ax=axes[1])
#         axes[1].set_title("Visual Acuity – Right")
#         axes[1].tick_params(axis="x", rotation=45)
#     else:
#         axes[1].axis("off")
#     plt.tight_layout()
#     save_chart("visual_acuity", fig)


# def chart_pharmacy(pharmacy, drugs):
#     if pharmacy is not None and not pharmacy.empty and "Drug Name" in pharmacy.columns:
#         top_drugs = pharmacy["Drug Name"].value_counts().head(15)
#         fig, ax = plt.subplots(figsize=(12, 6))
#         sns.barplot(y=top_drugs.index, x=top_drugs.values, palette="YlOrRd", ax=ax)
#         ax.set_title("Top 15 Prescribed Drugs")
#         plt.tight_layout()
#         save_chart("top15_drugs", fig)

#     if (pharmacy is not None and not pharmacy.empty
#             and "Dispensed" in pharmacy.columns):
#         fig, ax = plt.subplots(figsize=(6, 6))
#         pharmacy["Dispensed"].value_counts().plot(kind="pie", autopct="%1.1f%%", ax=ax)
#         ax.set_title("Pharmacy Dispense Rate")
#         ax.set_ylabel("")
#         plt.tight_layout()
#         save_chart("dispense_rate", fig)

#     if (drugs is not None and not drugs.empty
#             and {"Quantity", "Reorder Level", "Category"}.issubset(drugs.columns)):
#         d = drugs.copy()
#         d["Stock Status"] = np.where(d["Quantity"] <= d["Reorder Level"], "Reorder", "OK")
#         fig, ax = plt.subplots(figsize=(10, 5))
#         sns.barplot(data=d, x="Category", y="Quantity", hue="Stock Status", ax=ax)
#         ax.set_title("Drug Stock by Category")
#         plt.xticks(rotation=45)
#         plt.tight_layout()
#         save_chart("stock_by_category", fig)


# def chart_patient_journey(patient, consultations, lab_tests, pharmacy, optical):
#     def _n(df, col="Patient Id", fallback="Id"):
#         if df is None or df.empty:
#             return 0
#         if col in df.columns:
#             return df[col].nunique()
#         if fallback in df.columns:
#             return df[fallback].nunique()
#         return 0

#     stages = ["Registered", "Consulted", "Lab Tested", "Pharmacy", "Optical"]
#     counts = [
#         _n(patient, "Id", "Id"),
#         _n(consultations),
#         _n(lab_tests),
#         _n(pharmacy),
#         _n(optical),
#     ]
#     fig, ax = plt.subplots(figsize=(10, 5))
#     sns.barplot(x=stages, y=counts, palette="viridis", ax=ax)
#     ax.set_title("Patient Care Pathway Funnel")
#     ax.set_ylabel("Unique Patients")
#     for i, v in enumerate(counts):
#         ax.text(i, v + 0.5, str(v), ha="center")
#     plt.tight_layout()
#     save_chart("patient_journey_funnel", fig)


# # ============================================================
# # Entry Point
# # ============================================================
# def run():
#     print("\n=== COREP ANALYSIS STARTED ===")
#     data = clean_all(load_data())

#     # Report availability
#     print("\n📊 Data availability:")
#     for k, df in data.items():
#         status = "✅" if (df is not None and not df.empty) else "❌ empty"
#         n = 0 if df is None else len(df)
#         print(f"   {k:15s} {status}  ({n} rows)")

#     print("\n📊 Generating charts ...")
#     jobs = [
#         ("patient_demographics", lambda: chart_patient_demographics(data["patient"])),
#         ("registration_trends",  lambda: chart_registration_trends(data["patient"])),
#         ("vitals",               lambda: chart_vitals(data["nursing"])),
#         ("consultations",        lambda: chart_consultations(data["consultations"], data["patient"])),
#         ("lab_tests",            lambda: chart_lab_tests(data["lab_tests"])),
#         ("optical",              lambda: chart_optical(data["optical"])),
#         ("pharmacy",             lambda: chart_pharmacy(data["pharmacy"], data["drugs"])),
#         ("patient_journey",      lambda: chart_patient_journey(
#                                         data["patient"], data["consultations"],
#                                         data["lab_tests"], data["pharmacy"], data["optical"])),
#     ]
#     for name, fn in jobs:
#         try:
#             fn()
#         except Exception as e:
#             print(f"  ⚠️  {name} failed: {e}")

#     print(f"\n✅ All charts saved to ./{CHART_DIR}/  ({_chart_counter['n']} files)")
#     return data


# if __name__ == "__main__":
#     run()