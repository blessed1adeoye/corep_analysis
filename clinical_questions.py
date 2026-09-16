"""
Answers specific clinical questions from COREP data
and saves dedicated charts to ./charts/.
"""

import os
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

CHART_DIR = "charts"
os.makedirs(CHART_DIR, exist_ok=True)


def _merge_cons_patient(consultations, patient):
    return consultations.merge(
        patient[["Id", "Age", "Gender", "Age Group"]],
        left_on="Patient Id", right_on="Id",
        suffixes=("", "_pat"), how="left")


# Q1: Which diagnosis is most common in children (<=12)?
def q_common_diagnosis_children(consultations, patient):
    df = _merge_cons_patient(consultations, patient)
    children = df[df["Age"] <= 12]
    if children.empty:
        print("Q1: No children data."); return None
    top = children["Diagnosis"].value_counts().head(10)
    fig, ax = plt.subplots(figsize=(11, 6))
    sns.barplot(y=top.index, x=top.values, palette="rocket", ax=ax)
    ax.set_title("Top Diagnoses in Children (≤12 yrs)")
    plt.tight_layout()
    fig.savefig(os.path.join(CHART_DIR, "Q1_children_diagnoses.png"), dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"Q1 → Most common in children: {top.index[0]} ({top.iloc[0]} cases)")
    return top


# Q2: Which gender has the highest admission rate?
def q_admission_by_gender(patient):
    if "Is Admitted" not in patient.columns or "Gender" not in patient.columns:
        return None
    p = patient.copy()
    p["Is Admitted"] = p["Is Admitted"].astype(str).str.lower().isin(["true", "yes", "1"])
    rate = p.groupby("Gender")["Is Admitted"].mean().mul(100).round(1)
    fig, ax = plt.subplots(figsize=(8, 5))
    rate.plot(kind="bar", color="cornflowerblue", ax=ax)
    ax.set_ylabel("Admission Rate (%)")
    ax.set_title("Admission Rate by Gender")
    plt.tight_layout()
    fig.savefig(os.path.join(CHART_DIR, "Q2_admission_by_gender.png"), dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"Q2 → Admission rate by gender: {rate.to_dict()}")
    return rate


# Q3: Which drugs are most prescribed for malaria?
def q_drugs_for_malaria(consultations, pharmacy):
    mal = consultations[consultations["Diagnosis"].astype(str).str.contains(
        "malaria", case=False, na=False)]
    if mal.empty:
        print("Q3: No malaria diagnoses."); return None
    merged = pharmacy.merge(mal[["Id", "Patient Id"]],
                            left_on="Consultation Id", right_on="Id",
                            suffixes=("", "_cons"))
    top = merged["Drug Name"].value_counts().head(10)
    fig, ax = plt.subplots(figsize=(11, 6))
    sns.barplot(y=top.index, x=top.values, palette="flare", ax=ax)
    ax.set_title("Top Drugs Prescribed for Malaria")
    plt.tight_layout()
    fig.savefig(os.path.join(CHART_DIR, "Q3_malaria_drugs.png"), dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"Q3 → Top malaria drug: {top.index[0] if len(top) else 'N/A'}")
    return top


# Q4: Which age group has the highest malaria positive rate?
def q_malaria_by_age(lab_tests, patient):
    if "Malaria Parasite" not in lab_tests.columns:
        return None
    df = lab_tests.merge(patient[["Id", "Age Group"]],
                         left_on="Patient Id", right_on="Id", how="left")
    df["Positive"] = df["Malaria Parasite"].astype(str).str.lower().isin(
        ["positive", "pos", "reactive", "+", "1", "true"])
    rate = df.groupby("Age Group")["Positive"].mean().mul(100).round(1)
    fig, ax = plt.subplots(figsize=(10, 5))
    rate.plot(kind="bar", color="crimson", ax=ax)
    ax.set_ylabel("Malaria Positive Rate (%)")
    ax.set_title("Malaria Positive Rate by Age Group")
    plt.tight_layout()
    fig.savefig(os.path.join(CHART_DIR, "Q4_malaria_by_age.png"), dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"Q4 → Malaria positivity by age: {rate.to_dict()}")
    return rate


# Q5: What is the average time from consultation to lab test?
def q_consult_to_lab_time(consultations, lab_tests):
    if consultations.empty or lab_tests.empty:
        return None
    if "Consultation Id" not in lab_tests.columns:
        return None
    df = lab_tests.merge(consultations[["Id", "Created At"]],
                         left_on="Consultation Id", right_on="Id",
                         suffixes=("", "_cons"))
    delta = (df["Created At"] - df["Created At_cons"]).dt.total_seconds() / 60  # min
    delta = delta.dropna()
    if delta.empty:
        return None
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.histplot(delta, bins=30, kde=True, color="teal", ax=ax)
    ax.set_xlabel("Minutes")
    ax.set_title("Time from Consultation → Lab Test")
    plt.tight_layout()
    fig.savefig(os.path.join(CHART_DIR, "Q5_consult_to_lab_time.png"), dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"Q5 → Median consult→lab time: {delta.median():.1f} min")
    return delta


# Q6: Which staff member completed the most consultations?
def q_staff_workload(consultations):
    if "Created By Id" not in consultations.columns:
        return None
    top = consultations["Created By Id"].value_counts().head(15)
    fig, ax = plt.subplots(figsize=(12, 5))
    top.plot(kind="bar", color="teal", ax=ax)
    ax.set_title("Top Staff by Consultation Volume")
    plt.tight_layout()
    fig.savefig(os.path.join(CHART_DIR, "Q6_staff_workload.png"), dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"Q6 → Top staff: {top.index[0]} ({top.iloc[0]} consultations)")
    return top


# Q7: What is the average length of stay for admitted patients?
def q_los(patient):
    if "Is Admitted" not in patient.columns or "Updated At" not in patient.columns:
        return None
    p = patient.copy()
    p["Is Admitted"] = p["Is Admitted"].astype(str).str.lower().isin(["true", "yes", "1"])
    adm = p[p["Is Admitted"]]
    if adm.empty:
        return None
    los = (adm["Updated At"] - adm["Created At"]).dt.days
    los = los.dropna()
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.histplot(los, bins=20, kde=True, color="darkorange", ax=ax)
    ax.set_title("Length of Stay (days) – Admitted Patients")
    plt.tight_layout()
    fig.savefig(os.path.join(CHART_DIR, "Q7_length_of_stay.png"), dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"Q7 → Mean LOS: {los.mean():.1f} days | Median: {los.median():.1f} days")
    return los


def run_all(consultations, patient, lab_tests, pharmacy):
    print("\n=== CLINICAL QUESTIONS ===")
    q_common_diagnosis_children(consultations, patient)
    q_admission_by_gender(patient)
    q_drugs_for_malaria(consultations, pharmacy)
    q_malaria_by_age(lab_tests, patient)
    q_consult_to_lab_time(consultations, lab_tests)
    q_staff_workload(consultations)
    q_los(patient)


if __name__ == "__main__":
    import main_analysis as analysis
    d = analysis.clean_all(analysis.load_data())
    run_all(d["consultations"], d["patient"], d["lab_tests"], d["pharmacy"])