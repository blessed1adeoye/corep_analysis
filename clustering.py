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

from main_analysis import filter_real_drugs

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
    fig.savefig(os.path.join(CHART_DIR, "Q1_children_diagnoses.png"),
                dpi=110, bbox_inches="tight")
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
    fig.savefig(os.path.join(CHART_DIR, "Q2_admission_by_gender.png"),
                dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"Q2 → Admission rate by gender: {rate.to_dict()}")
    return rate


# Q3: Which drugs are most dispensed for malaria? (placeholders filtered)
def q_drugs_for_malaria(consultations, pharmacy):
    mal = consultations[consultations["Diagnosis"].astype(str).str.contains(
        "malaria", case=False, na=False)]
    if mal.empty:
        print("Q3: No malaria diagnoses."); return None

    merged = pharmacy.merge(mal[["Id", "Patient Id"]],
                            left_on="Consultation Id", right_on="Id",
                            suffixes=("", "_cons"))

    names = filter_real_drugs(merged["Drug Name"])
    top = names.value_counts().head(10)
    if top.empty:
        print("Q3: No valid drug names after filtering.")
        return None

    fig, ax = plt.subplots(figsize=(11, 6))
    sns.barplot(y=top.index, x=top.values, palette="flare", ax=ax)
    ax.set_title("Top Drugs Dispensed for Malaria")
    ax.set_xlabel("Number of Dispensations")
    plt.tight_layout()
    fig.savefig(os.path.join(CHART_DIR, "Q3_malaria_drugs.png"),
                dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"Q3 → Top malaria drug: {top.index[0]}")
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
    fig.savefig(os.path.join(CHART_DIR, "Q4_malaria_by_age.png"),
                dpi=110, bbox_inches="tight")
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
    delta = (df["Created At"] - df["Created At_cons"]).dt.total_seconds() / 60
    delta = delta.dropna()
    if delta.empty:
        return None
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.histplot(delta, bins=30, kde=True, color="teal", ax=ax)
    ax.set_xlabel("Minutes")
    ax.set_title("Time from Consultation → Lab Test")
    plt.tight_layout()
    fig.savefig(os.path.join(CHART_DIR, "Q5_consult_to_lab_time.png"),
                dpi=110, bbox_inches="tight")
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
    fig.savefig(os.path.join(CHART_DIR, "Q6_staff_workload.png"),
                dpi=110, bbox_inches="tight")
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
    fig.savefig(os.path.join(CHART_DIR, "Q7_length_of_stay.png"),
                dpi=110, bbox_inches="tight")
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
    
# # Patient Segmentation (K-Means)



# """
# Patient clustering:
#   - Segment patients by symptom/diagnosis patterns + demographics.
#   - K-Means with elbow + silhouette.
#   - Interprets each cluster.
# """

# import os
# import warnings
# warnings.filterwarnings("ignore")

# import numpy as np
# import pandas as pd
# import matplotlib
# matplotlib.use("Agg")
# import matplotlib.pyplot as plt
# import seaborn as sns

# from sklearn.preprocessing import StandardScaler
# from sklearn.cluster import KMeans
# from sklearn.decomposition import PCA
# from sklearn.metrics import silhouette_score
# from sklearn.feature_extraction.text import TfidfVectorizer

# CHART_DIR = "charts"
# REPORT_DIR = "reports"
# os.makedirs(CHART_DIR, exist_ok=True)
# os.makedirs(REPORT_DIR, exist_ok=True)


# def _save(fig, name):
#     path = os.path.join(CHART_DIR, name)
#     fig.savefig(path, dpi=110, bbox_inches="tight")
#     plt.close(fig)
#     print(f"  💾 {path}")


# def build_features(patient, consultations, nursing):
#     """Build one row per patient with demographics + clinical aggregates + symptom bag."""
#     p = patient.copy()
#     if "Age" not in p.columns and "Date Of Birth" in p.columns:
#         p["Date Of Birth"] = pd.to_datetime(p["Date Of Birth"], errors="coerce")
#         p["Age"] = ((pd.Timestamp.today() - p["Date Of Birth"]).dt.days / 365.25).round(1)

#     # Consultation aggregates
#     if not consultations.empty and "Patient Id" in consultations.columns:
#         agg = consultations.groupby("Patient Id").agg(
#             n_consultations=("Id", "count"),
#             n_unique_diagnoses=("Diagnosis", "nunique"),
#             n_referrals=("Refer To Pharmacy", lambda s: s.astype(str).str.lower().isin(
#                 ["true", "yes", "1"]).sum()),
#             diagnoses=("Diagnosis", lambda x: " | ".join(x.dropna().astype(str))),
#         )
#         p = p.merge(agg, left_on="Id", right_index=True, how="left")

#     # Vitals
#     if not nursing.empty and "Patient Id" in nursing.columns:
#         vit = nursing.groupby("Patient Id").agg(
#             avg_sys=("Blood Pressure Systolic", "mean"),
#             avg_dia=("Blood Pressure Diastolic", "mean"),
#             avg_temp=("Temperature", "mean"),
#             avg_pulse=("Pulse Rate", "mean"),
#             avg_spo2=("Oxygen Saturation", "mean"),
#         )
#         p = p.merge(vit, left_on="Id", right_index=True, how="left")

#     # Fill numerics
#     num_cols = ["Age", "n_consultations", "n_unique_diagnoses", "n_referrals",
#                 "avg_sys", "avg_dia", "avg_temp", "avg_pulse", "avg_spo2"]
#     for c in num_cols:
#         if c in p.columns:
#             p[c] = pd.to_numeric(p[c], errors="coerce").fillna(0)

#     # TF-IDF on diagnosis text
#     if "diagnoses" in p.columns:
#         text = p["diagnoses"].fillna("").astype(str)
#         if text.str.strip().ne("").any():
#             tfidf = TfidfVectorizer(max_features=25, stop_words="english")
#             X_text = tfidf.fit_transform(text).toarray()
#             text_df = pd.DataFrame(X_text, columns=[f"dx_{w}" for w in tfidf.get_feature_names_out()],
#                                     index=p.index)
#             p = pd.concat([p, text_df], axis=1)

#     # Gender one-hot
#     if "Gender" in p.columns:
#         p = pd.concat([p, pd.get_dummies(p["Gender"], prefix="g", drop_first=True)], axis=1)

#     return p


# def find_best_k(X, k_range=range(2, 9)):
#     inertias, sil = [], []
#     for k in k_range:
#         km = KMeans(n_clusters=k, random_state=42, n_init=10)
#         labels = km.fit_predict(X)
#         inertias.append(km.inertia_)
#         sil.append(silhouette_score(X, labels))
#     return list(k_range), inertias, sil


# def run(patient, consultations, nursing):
#     print("\n=== CLUSTERING MODULE ===")

#     df = build_features(patient, consultations, nursing)
#     if df.empty or len(df) < 20:
#         print("  ⚠️  Not enough patients to cluster.")
#         return

#     feature_cols = [c for c in df.columns
#                     if c.startswith(("dx_", "g_")) or c in
#                     ["Age", "n_consultations", "n_unique_diagnoses", "n_referrals",
#                      "avg_sys", "avg_dia", "avg_temp", "avg_pulse", "avg_spo2"]]
#     X_raw = df[feature_cols].fillna(0).values
#     X = StandardScaler().fit_transform(X_raw)

#     # Elbow + silhouette
#     ks, inertias, sil = find_best_k(X)
#     fig, axes = plt.subplots(1, 2, figsize=(14, 5))
#     axes[0].plot(ks, inertias, marker="o")
#     axes[0].set_title("Elbow Method"); axes[0].set_xlabel("k"); axes[0].set_ylabel("Inertia")
#     axes[1].plot(ks, sil, marker="o", color="green")
#     axes[1].set_title("Silhouette Score"); axes[1].set_xlabel("k"); axes[1].set_ylabel("Score")
#     plt.tight_layout()
#     _save(fig, "cluster_elbow_silhouette.png")

#     best_k = ks[int(np.argmax(sil))]
#     print(f"  🎯 Best k = {best_k} (silhouette = {max(sil):.3f})")

#     km = KMeans(n_clusters=best_k, random_state=42, n_init=10)
#     df["Cluster"] = km.fit_predict(X)

#     # PCA 2D visualization
#     pcs = PCA(n_components=2).fit_transform(X)
#     df["PC1"], df["PC2"] = pcs[:, 0], pcs[:, 1]
#     fig, ax = plt.subplots(figsize=(10, 7))
#     sns.scatterplot(data=df, x="PC1", y="PC2", hue="Cluster",
#                     palette="tab10", s=60, ax=ax)
#     ax.set_title(f"Patient Clusters (k={best_k}) – PCA projection")
#     plt.tight_layout()
#     _save(fig, "cluster_pca_scatter.png")

#     # Cluster profile
#     profile_cols = [c for c in ["Age", "n_consultations", "n_unique_diagnoses",
#                                  "n_referrals", "avg_sys", "avg_temp"] if c in df.columns]
#     profile = df.groupby("Cluster")[profile_cols].mean().round(2)
#     profile["size"] = df["Cluster"].value_counts().sort_index()
#     print("\n  Cluster profile:\n", profile)
#     profile.to_csv(os.path.join(REPORT_DIR, "cluster_profile.csv"))

#     # Heatmap
#     fig, ax = plt.subplots(figsize=(10, 5))
#     sns.heatmap(profile[profile_cols].T, annot=True, cmap="YlGnBu", fmt=".1f", ax=ax)
#     ax.set_title("Cluster Profiles (mean values)")
#     plt.tight_layout()
#     _save(fig, "cluster_heatmap.png")

#     # Top diagnosis per cluster
#     if "diagnoses" in df.columns:
#         print("\n  Top diagnosis per cluster:")
#         for c in sorted(df["Cluster"].unique()):
#             txt = " ".join(df[df["Cluster"] == c]["diagnoses"].fillna("").astype(str))
#             words = pd.Series(txt.split("|")).str.strip().value_counts().head(3)
#             print(f"    Cluster {c}: {list(words.index)}")

#     df[["Id", "Cluster"]].to_csv(os.path.join(REPORT_DIR, "patient_clusters.csv"), index=False)
#     print("  ✅ Patient cluster assignments saved.")


# if __name__ == "__main__":
#     import main_analysis as analysis
#     d = analysis.clean_all(analysis.load_data())
#     run(d["patient"], d["consultations"], d["nursing"])