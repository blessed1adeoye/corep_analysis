"""
COREP Data Analysis
- Loads all sheets
- Cleans data
- Generates and SAVES every chart to ./charts/
"""

import os
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


def save_chart(name, fig=None):
    """Save current or given figure and close it."""
    fig = fig or plt.gcf()
    _chart_counter["n"] += 1
    filename = f"{_chart_counter['n']:02d}_{name}.png"
    path = os.path.join(CHART_DIR, filename)
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"  💾 Saved → {path}")
    return path


# ---------------- Load ----------------
def load_data(path=EXCEL_FILE):
    sheets = {
        "patient": "patient",
        "nursing": "Nursing_Assessments",
        "consultations": "Consultations",
        "lab_tests": "Lab_Tests",
        "optical": "Optical_Assessments",
        "pharmacy": "Pharmacy_Orders",
        "drugs": "Drugs",
    }
    data = {}
    for key, sheet in sheets.items():
        try:
            data[key] = pd.read_excel(path, sheet_name=sheet)
            print(f"✅ Loaded {sheet:25s} shape={data[key].shape}")
        except Exception as e:
            print(f"⚠️  Could not load {sheet}: {e}")
            data[key] = pd.DataFrame()
    return data


def clean(df, date_cols=()):
    df = df.copy()
    df.columns = df.columns.astype(str).str.strip()
    for c in date_cols:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors="coerce")
    return df.dropna(how="all")


def clean_all(data):
    data["patient"] = clean(data["patient"], ["Created At", "Updated At", "Date Of Birth"])
    data["nursing"] = clean(data["nursing"], ["Created At", "Updated At", "Completed At"])
    data["consultations"] = clean(data["consultations"], ["Created At", "Updated At", "Completed At"])
    data["lab_tests"] = clean(data["lab_tests"], ["Created At", "Updated At", "Completed At", "Viewed At"])
    data["optical"] = clean(data["optical"],
                            ["Created At", "Updated At", "Completed At", "Viewed At", "Viewed By Physician At"])
    data["pharmacy"] = clean(data["pharmacy"], ["Created At", "Updated At", "Dispensed At"])
    data["drugs"] = clean(data["drugs"], ["Created At", "Updated At"])

    # Derived: age
    if "Date Of Birth" in data["patient"].columns:
        today = pd.Timestamp.today()
        data["patient"]["Age"] = ((today - data["patient"]["Date Of Birth"]).dt.days / 365.25).round(1)
        bins = [0, 5, 12, 18, 35, 50, 65, 120]
        labels = ["0-5", "6-12", "13-18", "19-35", "36-50", "51-65", "65+"]
        data["patient"]["Age Group"] = pd.cut(data["patient"]["Age"], bins=bins, labels=labels)
    return data


# ---------------- Charts ----------------
def chart_patient_demographics(patient):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    if "Gender" in patient.columns:
        patient["Gender"].value_counts().plot(kind="bar", ax=axes[0, 0], color="steelblue")
        axes[0, 0].set_title("Gender Distribution")

    if "Age" in patient.columns:
        sns.histplot(patient["Age"].dropna(), bins=20, kde=True, ax=axes[0, 1], color="teal")
        axes[0, 1].set_title("Age Distribution")

    if "Age Group" in patient.columns:
        patient["Age Group"].value_counts().sort_index().plot(kind="bar", ax=axes[1, 0], color="coral")
        axes[1, 0].set_title("Age Group Distribution")

    if "Is Admitted" in patient.columns:
        patient["Is Admitted"].value_counts().plot(
            kind="pie", autopct="%1.1f%%", ax=axes[1, 1], colors=["#66b3ff", "#ff9999"])
        axes[1, 1].set_title("Admission Status")
        axes[1, 1].set_ylabel("")

    plt.tight_layout()
    save_chart("patient_demographics", fig)


def chart_registration_trends(patient):
    if "Created At" not in patient.columns or patient["Created At"].isna().all():
        return
    patient = patient.copy()
    patient["Month"] = patient["Created At"].dt.to_period("M").astype(str)
    monthly = patient.groupby("Month").size()

    fig, ax = plt.subplots(figsize=(14, 5))
    monthly.plot(kind="line", marker="o", color="navy", ax=ax)
    ax.set_title("Monthly Patient Registrations")
    ax.set_ylabel("New Patients")
    plt.tight_layout()
    save_chart("monthly_registrations", fig)

    patient["DayOfWeek"] = patient["Created At"].dt.day_name()
    order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.countplot(data=patient, x="DayOfWeek", order=order, palette="viridis", ax=ax)
    ax.set_title("Registrations by Day of Week")
    plt.xticks(rotation=45)
    plt.tight_layout()
    save_chart("registrations_by_dayofweek", fig)


def chart_vitals(nursing):
    vitals = ["Blood Pressure Systolic", "Blood Pressure Diastolic",
              "Pulse Rate", "Temperature", "Respiratory Rate", "Oxygen Saturation"]
    vitals = [c for c in vitals if c in nursing.columns]
    if not vitals:
        return

    # Box plots
    fig, axes = plt.subplots(2, 3, figsize=(16, 8))
    for ax, col in zip(axes.flatten(), vitals):
        sns.boxplot(y=nursing[col].dropna(), ax=ax, color="lightblue")
        ax.set_title(col)
    for ax in axes.flatten()[len(vitals):]:
        ax.axis("off")
    plt.tight_layout()
    save_chart("vitals_boxplots", fig)

    # Distributions
    fig, axes = plt.subplots(2, 3, figsize=(16, 8))
    for ax, col in zip(axes.flatten(), vitals):
        sns.histplot(nursing[col].dropna(), kde=True, ax=ax, color="seagreen")
        ax.set_title(f"Distribution – {col}")
    for ax in axes.flatten()[len(vitals):]:
        ax.axis("off")
    plt.tight_layout()
    save_chart("vitals_distributions", fig)

    # Correlation heatmap
    fig, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(nursing[vitals].corr(), annot=True, cmap="coolwarm", fmt=".2f", ax=ax)
    ax.set_title("Vitals Correlation Matrix")
    plt.tight_layout()
    save_chart("vitals_correlation", fig)

    # Biohazard / isolation
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    if "Biohazard Risk" in nursing.columns:
        nursing["Biohazard Risk"].value_counts().plot(kind="bar", ax=axes[0], color="tomato")
        axes[0].set_title("Biohazard Risk")
    if "Isolation Required" in nursing.columns:
        nursing["Isolation Required"].value_counts().plot(kind="bar", ax=axes[1], color="orange")
        axes[1].set_title("Isolation Required")
    plt.tight_layout()
    save_chart("biohazard_isolation", fig)


def chart_consultations(consultations, patient):
    if consultations.empty or "Diagnosis" not in consultations.columns:
        return

    top_diag = consultations["Diagnosis"].value_counts().head(15)
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.barplot(y=top_diag.index, x=top_diag.values, palette="magma", ax=ax)
    ax.set_title("Top 15 Diagnoses")
    plt.tight_layout()
    save_chart("top15_diagnoses", fig)

    # Merge with patient to enrich
    cons_p = consultations.merge(
        patient[["Id", "Age", "Gender", "Age Group"]],
        left_on="Patient Id", right_on="Id", suffixes=("", "_pat"), how="left")

    if "Gender" in cons_p.columns:
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

    if "Created At" in consultations.columns:
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
    if lab_tests.empty:
        return
    lab_cols = [c for c in ["Malaria Parasite", "Random Blood Sugar", "Hbsag"] if c in lab_tests.columns]
    if lab_cols:
        fig, axes = plt.subplots(1, len(lab_cols), figsize=(6 * len(lab_cols), 4))
        if len(lab_cols) == 1:
            axes = [axes]
        for ax, col in zip(axes, lab_cols):
            lab_tests[col].value_counts(dropna=False).plot(kind="bar", ax=ax, color="indianred")
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
    if optical.empty:
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
        optical["Glasses Allocated"].value_counts().plot(kind="bar", color="mediumpurple", ax=ax)
        ax.set_title("Glasses Allocated")
        plt.tight_layout()
        save_chart("glasses_allocated", fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    if "Visual Acuity Left" in optical.columns:
        sns.countplot(data=optical, x="Visual Acuity Left", palette="Blues", ax=axes[0])
        axes[0].set_title("Visual Acuity – Left")
        axes[0].tick_params(axis="x", rotation=45)
    if "Visual Acuity Right" in optical.columns:
        sns.countplot(data=optical, x="Visual Acuity Right", palette="Greens", ax=axes[1])
        axes[1].set_title("Visual Acuity – Right")
        axes[1].tick_params(axis="x", rotation=45)
    plt.tight_layout()
    save_chart("visual_acuity", fig)


def chart_pharmacy(pharmacy, drugs):
    if not pharmacy.empty and "Drug Name" in pharmacy.columns:
        top_drugs = pharmacy["Drug Name"].value_counts().head(15)
        fig, ax = plt.subplots(figsize=(12, 6))
        sns.barplot(y=top_drugs.index, x=top_drugs.values, palette="YlOrRd", ax=ax)
        ax.set_title("Top 15 Prescribed Drugs")
        plt.tight_layout()
        save_chart("top15_drugs", fig)

    if not pharmacy.empty and "Dispensed" in pharmacy.columns:
        fig, ax = plt.subplots(figsize=(6, 6))
        pharmacy["Dispensed"].value_counts().plot(kind="pie", autopct="%1.1f%%", ax=ax)
        ax.set_title("Pharmacy Dispense Rate")
        ax.set_ylabel("")
        plt.tight_layout()
        save_chart("dispense_rate", fig)

    if not drugs.empty and {"Quantity", "Reorder Level", "Category"}.issubset(drugs.columns):
        drugs = drugs.copy()
        drugs["Stock Status"] = np.where(drugs["Quantity"] <= drugs["Reorder Level"], "Reorder", "OK")
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.barplot(data=drugs, x="Category", y="Quantity", hue="Stock Status", ax=ax)
        ax.set_title("Drug Stock by Category")
        plt.xticks(rotation=45)
        plt.tight_layout()
        save_chart("stock_by_category", fig)


def chart_patient_journey(patient, consultations, lab_tests, pharmacy, optical):
    stages = ["Registered", "Consulted", "Lab Tested", "Pharmacy", "Optical"]
    counts = [
        patient["Id"].nunique() if "Id" in patient.columns else 0,
        consultations["Patient Id"].nunique() if "Patient Id" in consultations.columns else 0,
        lab_tests["Patient Id"].nunique() if "Patient Id" in lab_tests.columns else 0,
        pharmacy["Patient Id"].nunique() if "Patient Id" in pharmacy.columns else 0,
        optical["Patient Id"].nunique() if "Patient Id" in optical.columns else 0,
    ]
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(x=stages, y=counts, palette="viridis", ax=ax)
    ax.set_title("Patient Care Pathway Funnel")
    ax.set_ylabel("Unique Patients")
    for i, v in enumerate(counts):
        ax.text(i, v + 0.5, str(v), ha="center")
    plt.tight_layout()
    save_chart("patient_journey_funnel", fig)


# ---------------- Entrypoint ----------------
def run():
    print("\n=== COREP ANALYSIS STARTED ===")
    data = clean_all(load_data())

    print("\n📊 Generating charts ...")
    chart_patient_demographics(data["patient"])
    chart_registration_trends(data["patient"])
    chart_vitals(data["nursing"])
    chart_consultations(data["consultations"], data["patient"])
    chart_lab_tests(data["lab_tests"])
    chart_optical(data["optical"])
    chart_pharmacy(data["pharmacy"], data["drugs"])
    chart_patient_journey(data["patient"], data["consultations"],
                          data["lab_tests"], data["pharmacy"], data["optical"])

    print(f"\n✅ All charts saved to ./{CHART_DIR}/  ({_chart_counter['n']} files)")
    return data


if __name__ == "__main__":
    run()