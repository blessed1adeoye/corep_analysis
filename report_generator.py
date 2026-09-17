"""
COREP Report Generator — FULL chart set + branding + palette-safe
- Responsive HTML report (mobile/desktop)
- Microsoft Word (.docx) — landscape, 0.5" x 0.5" margins, big colorful charts
- Optional PDF via WeasyPrint (graceful skip)
- Prescription-stripped Treatment Plans
- Placeholder drug-name filtering
- Branding on header, footer, and title page
- Palette safety net (Plotly names auto-mapped to seaborn)
"""

import os
import re
import base64
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from jinja2 import Template

from branding import (
    BRAND_LINE, BRAND_LINE_FULL,
    DEVELOPER_LINE, DEVELOPER_LINE_FULL, DEVELOPER_LINE_SHORT,
    COPYRIGHT, POWERED_BY, TRADEMARK,
)
from main_analysis import filter_real_drugs

warnings.filterwarnings("ignore")
sns.set_style("whitegrid")

CHART_DIR = "charts"
REPORT_DIR = "reports"
DOCX_CHART_DIR = os.path.join(REPORT_DIR, "_docx_charts")
os.makedirs(REPORT_DIR, exist_ok=True)
os.makedirs(DOCX_CHART_DIR, exist_ok=True)


# ============================================================
# Palette safety (Plotly → seaborn mapping)
# ============================================================
_PALETTE_MAP = {
    "Vivid":     "husl",
    "Bold":      "Set1",
    "Prism":     "Set2",
    "Pastel":    "pastel",
    "Pastel1":   "pastel",
    "Pastel2":   "pastel",
    "Blues_d":   "Blues",
    "Greens_d":  "Greens",
    "Reds_d":    "Reds",
    "Oranges_d": "Oranges",
    "Purples_d": "Purples",
    "YlOrRd_d":  "YlOrRd",
    "YlOrBr_d":  "YlOrBr",
    "Dark24":    "tab20",
    "Light24":   "tab20b",
    "Alphabet":  "husl",
    "Antique":   "Set2",
    "Astral":    "husl",
    "D3":        "tab10",
    "T10":       "tab10",
    "G10":       "tab10",
    "Plotly":    "Set1",
    "Safe":      "Set2",
    "Dark2":     "Dark2",
}


def _safe_palette(name):
    """Return a seaborn-safe palette name or list."""
    if name is None:
        return None
    if isinstance(name, (list, tuple)):
        return name
    return _PALETTE_MAP.get(str(name), str(name))


# ============================================================
# Prescription stripping
# ============================================================
PRESCRIPTION_PATTERNS = [
    r"\bRx\s*[:\-]?\s*[^.;\n]+",
    r"\bPrescription\s*[:\-]?\s*[^.;\n]+",
    r"\bPrescribed\s*[:\-]?\s*[^.;\n]+",
    r"\b(?:Tab|Tabs|Tablet|Tablets|Cap|Caps|Capsule|Capsules|"
    r"Syr|Syrup|Susp|Suspension|Inj|Injection|Cream|Oint|Ointment|"
    r"Drops|Spray|Inhaler|Supp|Suppository)"
    r"\.?\s+[A-Za-z][A-Za-z0-9\-]*(?:\s+[A-Za-z0-9\-]+)*"
    r"(?:\s+\d+\s*(?:mg|g|ml|mcg|IU|tab|caps?))?"
    r"(?:\s+(?:bd|tds|qid|od|nocte|prn|stat|daily|twice|thrice))?"
    r"(?:\s+(?:x|for)\s*\d+\s*(?:days?|weeks?|months?))?",
    r"\b\d+\s*(?:mg|g|ml|mcg|IU)\b[^.;\n]*",
    r"\b(?:bd|tds|qid|qds|od|nocte|prn|stat|q4h|q6h|q8h|q12h)\b",
    r"\b(?:x|for)\s*\d+\s*/\s*\d+\b",
    r"\bfor\s+\d+\s*(?:days?|weeks?|months?)\b",
    r"\bDispense[ds]?\b[^.;\n]*",
    r"\bRefer\s+to\s+(?:the\s+)?(?:Pharmacy|Pharmacist)\b[^.;\n]*",
    r"\bPharmacy\b[^.;\n]*",
    r"\bPharmacist\b[^.;\n]*",
]
DRUG_HINT_WORDS = (
    r"(?:Amoxicillin|Amoxycillin|Paracetamol|Acetaminophen|Ibuprofen|"
    r"Artemether|Lumefantrine|Artemether/Lumefantrine|ACT|Coartem|"
    r"Metronidazole|Ciprofloxacin|Azithromycin|Ceftriaxone|"
    r"ORS|Zinc|Vitamin\s*C|Folic\s*Acid|Ferrous\s*Sulphate|"
    r"Chloroquine|Quinine|Doxycycline|Albendazole|Mebendazole|"
    r"Amlodipine|Lisinopril|Losartan|Metformin|Glibenclamide|"
    r"Atorvastatin|Simvastatin|Omeprazole|Ranitidine|"
    r"Salbutamol|Prednisolone|Dexamethasone|Hydrocortisone|"
    r"Chlorpheniramine|Diphenhydramine|Promethazine|"
    r"Tetracycline|Erythromycin|Gentamicin|"
    r"Normal\s*Saline|Dextrose|Ringer)"
)
DRUG_PATTERN = re.compile(r"\b" + DRUG_HINT_WORDS + r"\b[^.;\n]*", re.IGNORECASE)
COMPILED_PATTERNS = [re.compile(p, re.IGNORECASE | re.MULTILINE)
                     for p in PRESCRIPTION_PATTERNS]


def strip_prescription(text, remove_drug_names=True):
    if pd.isna(text):
        return text
    s = str(text)
    if remove_drug_names:
        s = DRUG_PATTERN.sub("", s)
    for pat in COMPILED_PATTERNS:
        s = pat.sub("", s)
    s = re.sub(r"\s*[;\-,]\s*[;\-,]+", "; ", s)
    s = re.sub(r"\s{2,}", " ", s)
    s = re.sub(r"^[\s;,\-\.]+", "", s)
    s = re.sub(r"[\s;,\-\.]+$", "", s)
    s = re.sub(r"\b(?:and|or|then|with)\s*$", "", s, flags=re.IGNORECASE)
    return s.strip()


def clean_treatment_plans(df):
    if df is None or df.empty or "Treatment Plan" not in df.columns:
        return df
    df = df.copy()
    df["Treatment Plan (Clinical)"] = df["Treatment Plan"].apply(strip_prescription)
    return df


# ============================================================
# Helpers
# ============================================================
def img_to_base64(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def collect_charts():
    if not os.path.isdir(CHART_DIR):
        return []
    files = sorted(f for f in os.listdir(CHART_DIR) if f.lower().endswith(".png"))
    return [(f, os.path.join(CHART_DIR, f)) for f in files]


def pretty_name(filename):
    n = filename.rsplit(".", 1)[0]
    n = n.split("_", 1)[-1]
    return n.replace("_", " ").title()


def _style_ax(ax, title, xlabel=None, ylabel=None):
    ax.set_title(title, fontsize=18, fontweight="bold", color="#0b4a6f", pad=14)
    if xlabel: ax.set_xlabel(xlabel, fontsize=13, fontweight="bold")
    if ylabel: ax.set_ylabel(ylabel, fontsize=13, fontweight="bold")
    ax.tick_params(axis="both", labelsize=11)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)


def _save_docx(fig, name):
    path = os.path.join(DOCX_CHART_DIR, name)
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def _watermark(fig):
    try:
        fig.text(0.99, 0.01, BRAND_LINE, ha="right", va="bottom",
                 fontsize=8, color="#999", alpha=0.75, style="italic")
    except Exception:
        pass


# ============================================================
# DOCX-specific chart generator (22 charts)
# ============================================================
def render_docx_charts(data):
    out = []
    p = data.get("patient", pd.DataFrame())
    cons = data.get("consultations", pd.DataFrame())
    nursing = data.get("nursing", pd.DataFrame())
    lab = data.get("lab_tests", pd.DataFrame())
    optical = data.get("optical", pd.DataFrame())
    pharm = data.get("pharmacy", pd.DataFrame())
    drugs = data.get("drugs", pd.DataFrame())

    # ---- 1. Patient Demographics ----
    if not p.empty:
        fig, axes = plt.subplots(2, 2, figsize=(16, 11))
        fig.suptitle("Patient Demographics", fontsize=22, fontweight="bold",
                     color="#0b4a6f", y=1.00)

        if "Gender" in p.columns and p["Gender"].notna().any():
            vc = p["Gender"].value_counts()
            sns.barplot(x=vc.index, y=vc.values, palette=_safe_palette("husl"),
                        ax=axes[0, 0], edgecolor="black")
            _style_ax(axes[0, 0], "Gender Distribution", ylabel="Patients")
        else:
            axes[0, 0].axis("off")

        if "Age" in p.columns and p["Age"].notna().any():
            sns.histplot(p["Age"].dropna(), bins=20, kde=True,
                         color="#e63946", ax=axes[0, 1], edgecolor="white")
            _style_ax(axes[0, 1], "Age Distribution", xlabel="Age (years)", ylabel="Count")
        else:
            axes[0, 1].axis("off")

        if "Age Group" in p.columns and p["Age Group"].notna().any():
            vc = p["Age Group"].value_counts().sort_index()
            sns.barplot(x=vc.index, y=vc.values, palette=_safe_palette("viridis"),
                        ax=axes[1, 0], edgecolor="black")
            _style_ax(axes[1, 0], "Age Group Distribution", ylabel="Patients")
        else:
            axes[1, 0].axis("off")

        if "Is Admitted" in p.columns and p["Is Admitted"].notna().any():
            vc = p["Is Admitted"].value_counts()
            axes[1, 1].pie(vc.values, labels=vc.index, autopct="%1.1f%%",
                            colors=sns.color_palette("Set2"), startangle=90,
                            textprops={"fontsize": 13, "fontweight": "bold"})
            axes[1, 1].set_title("Admission Status", fontsize=18,
                                  fontweight="bold", color="#0b4a6f")
        else:
            axes[1, 1].axis("off")

        _watermark(fig)
        plt.tight_layout()
        out.append(("Patient Demographics", _save_docx(fig, "01_demographics.png")))

    # ---- 2. Patient Arrivals by Hour ----
    if not p.empty and "Created At" in p.columns and p["Created At"].notna().any():
        t = p.copy()
        t["_dt"] = pd.to_datetime(t["Created At"], errors="coerce")
        t["Hour"] = t["_dt"].dt.hour
        hr = t["Hour"].value_counts().sort_index()
        if not hr.empty:
            fig, ax = plt.subplots(figsize=(16, 7))
            sns.barplot(x=hr.index, y=hr.values, palette=_safe_palette("crest"),
                        ax=ax, edgecolor="black")
            _style_ax(ax, "Patient Arrivals by Hour of Day",
                      xlabel="Hour", ylabel="Registrations")
            _watermark(fig); plt.tight_layout()
            out.append(("Patient Arrivals by Hour",
                        _save_docx(fig, "02_arrivals_by_hour.png")))

    # ---- 3. Top 15 Diagnoses ----
    if not cons.empty and "Diagnosis" in cons.columns:
        top = cons["Diagnosis"].value_counts().head(15)
        if not top.empty:
            fig, ax = plt.subplots(figsize=(16, 9))
            sns.barplot(y=top.index, x=top.values, palette=_safe_palette("rocket"),
                        ax=ax, edgecolor="black")
            _style_ax(ax, "Top 15 Diagnoses",
                      xlabel="Number of Cases", ylabel="Diagnosis")
            _watermark(fig); plt.tight_layout()
            out.append(("Top 15 Diagnoses",
                        _save_docx(fig, "03_diagnoses.png")))

    # ---- 4. Diagnoses by Gender ----
    if (not cons.empty and not p.empty
            and "Diagnosis" in cons.columns and "Gender" in p.columns):
        merged = cons.merge(p[["Id", "Gender"]], left_on="Patient Id",
                             right_on="Id", how="left", suffixes=("", "_p"))
        top10 = cons["Diagnosis"].value_counts().head(10).index
        sub = merged[merged["Diagnosis"].isin(top10)]
        ct = sub.groupby(["Diagnosis", "Gender"]).size().reset_index(name="Count")
        if not ct.empty:
            fig, ax = plt.subplots(figsize=(16, 9))
            sns.barplot(data=ct, y="Diagnosis", x="Count", hue="Gender",
                        palette=_safe_palette("Set2"), ax=ax, edgecolor="black")
            _style_ax(ax, "Top 10 Diagnoses by Gender",
                      xlabel="Cases", ylabel="")
            _watermark(fig); plt.tight_layout()
            out.append(("Top 10 Diagnoses by Gender",
                        _save_docx(fig, "04_dx_by_gender.png")))

    # ---- 5. Diagnoses by Age Group ----
    if (not cons.empty and not p.empty
            and "Diagnosis" in cons.columns and "Age Group" in p.columns):
        merged = cons.merge(p[["Id", "Age Group"]], left_on="Patient Id",
                             right_on="Id", how="left", suffixes=("", "_p"))
        top10 = cons["Diagnosis"].value_counts().head(10).index
        sub = merged[merged["Diagnosis"].isin(top10)]
        ct = sub.groupby(["Diagnosis", "Age Group"]).size().reset_index(name="Count")
        if not ct.empty:
            fig, ax = plt.subplots(figsize=(16, 9))
            sns.barplot(data=ct, y="Diagnosis", x="Count", hue="Age Group",
                        palette=_safe_palette("husl"), ax=ax, edgecolor="black")
            _style_ax(ax, "Top 10 Diagnoses by Age Group",
                      xlabel="Cases", ylabel="")
            _watermark(fig); plt.tight_layout()
            out.append(("Top 10 Diagnoses by Age Group",
                        _save_docx(fig, "05_dx_by_age.png")))

    # ---- 6. Referral Distribution ----
    if not cons.empty:
        refs = [c for c in ["Refer To Pharmacy", "Refer To Laboratory",
                            "Refer To Optician", "Refer To Specialist"]
                if c in cons.columns]
        if refs:
            def pos(x):
                return int(x.sum()) if x.dtype == bool else \
                       int(x.astype(str).str.lower().isin(["true", "yes", "1"]).sum())
            counts = cons[refs].apply(pos)
            if not counts.empty:
                fig, ax = plt.subplots(figsize=(14, 7))
                sns.barplot(x=counts.index, y=counts.values,
                            palette=["#2a9d8f", "#e9c46a", "#f4a261", "#e76f51"],
                            ax=ax, edgecolor="black")
                _style_ax(ax, "Referral Distribution",
                          ylabel="Number of Referrals")
                plt.xticks(rotation=15)
                _watermark(fig); plt.tight_layout()
                out.append(("Referral Distribution",
                            _save_docx(fig, "06_referrals.png")))

    # ---- 7. Consultations by Hour ----
    if not cons.empty and "Created At" in cons.columns and cons["Created At"].notna().any():
        t = cons.copy()
        t["_dt"] = pd.to_datetime(t["Created At"], errors="coerce")
        t["Hour"] = t["_dt"].dt.hour
        hr = t["Hour"].value_counts().sort_index()
        if not hr.empty:
            fig, ax = plt.subplots(figsize=(16, 7))
            sns.barplot(x=hr.index, y=hr.values, palette=_safe_palette("YlOrBr"),
                        ax=ax, edgecolor="black")
            _style_ax(ax, "Consultations by Hour of Day",
                      xlabel="Hour", ylabel="Consultations")
            _watermark(fig); plt.tight_layout()
            out.append(("Consultations by Hour",
                        _save_docx(fig, "07_consult_hour.png")))

    # ---- 8. Staff Workload ----
    if not cons.empty and "Created By Id" in cons.columns:
        sw = cons["Created By Id"].value_counts().head(15)
        if not sw.empty:
            fig, ax = plt.subplots(figsize=(16, 7))
            sns.barplot(x=sw.index.astype(str), y=sw.values,
                        palette=_safe_palette("Blues"), ax=ax, edgecolor="black")
            _style_ax(ax, "Staff Workload – Consultations",
                      xlabel="Staff ID", ylabel="Consultations")
            plt.xticks(rotation=45)
            _watermark(fig); plt.tight_layout()
            out.append(("Staff Workload",
                        _save_docx(fig, "08_staff.png")))

    # ---- 9. Vital Signs ----
    if not nursing.empty:
        vitals = [c for c in ["Blood Pressure Systolic", "Blood Pressure Diastolic",
                              "Pulse Rate", "Temperature",
                              "Respiratory Rate", "Oxygen Saturation"]
                  if c in nursing.columns]
        if vitals:
            fig, axes = plt.subplots(2, 3, figsize=(18, 10))
            fig.suptitle("Vital Signs Distribution", fontsize=22,
                         fontweight="bold", color="#0b4a6f")
            palette = sns.color_palette("husl", len(vitals))
            for ax, col, color in zip(axes.flatten(), vitals, palette):
                sns.boxplot(y=nursing[col].dropna(), ax=ax, color=color,
                            width=0.5, linewidth=2)
                sns.stripplot(y=nursing[col].dropna(), ax=ax, color="black",
                              size=3, alpha=0.4, jitter=True)
                _style_ax(ax, col, ylabel="Value")
            for ax in axes.flatten()[len(vitals):]:
                ax.axis("off")
            _watermark(fig); plt.tight_layout()
            out.append(("Vital Signs", _save_docx(fig, "09_vitals.png")))

    # ---- 10. Vitals Correlation ----
    if not nursing.empty:
        vitals = [c for c in ["Blood Pressure Systolic", "Blood Pressure Diastolic",
                              "Pulse Rate", "Temperature",
                              "Respiratory Rate", "Oxygen Saturation"]
                  if c in nursing.columns]
        if len(vitals) >= 2:
            corr = nursing[vitals].corr().round(2)
            fig, ax = plt.subplots(figsize=(12, 10))
            sns.heatmap(corr, annot=True, cmap="RdBu_r", fmt=".2f",
                        vmin=-1, vmax=1, ax=ax, linewidths=0.5)
            ax.set_title("Vitals Correlation Matrix", fontsize=18,
                          fontweight="bold", color="#0b4a6f", pad=14)
            _watermark(fig); plt.tight_layout()
            out.append(("Vitals Correlation",
                        _save_docx(fig, "10_vitals_corr.png")))

    # ---- 11. Biohazard & Isolation ----
    if not nursing.empty:
        fig, axes = plt.subplots(1, 2, figsize=(16, 7))
        if "Biohazard Risk" in nursing.columns and nursing["Biohazard Risk"].notna().any():
            vc = nursing["Biohazard Risk"].value_counts()
            sns.barplot(x=vc.index.astype(str), y=vc.values,
                        palette=_safe_palette("Reds"), ax=axes[0], edgecolor="black")
            _style_ax(axes[0], "Biohazard Risk", ylabel="Count")
        else:
            axes[0].axis("off")
        if "Isolation Required" in nursing.columns and nursing["Isolation Required"].notna().any():
            vc = nursing["Isolation Required"].value_counts()
            sns.barplot(x=vc.index.astype(str), y=vc.values,
                        palette=_safe_palette("Oranges"), ax=axes[1], edgecolor="black")
            _style_ax(axes[1], "Isolation Required", ylabel="Count")
        else:
            axes[1].axis("off")
        _watermark(fig); plt.tight_layout()
        out.append(("Biohazard & Isolation",
                    _save_docx(fig, "11_biohazard.png")))

    # ---- 12. Lab Test Results ----
    if not lab.empty:
        lab_cols = [c for c in ["Malaria Parasite", "Random Blood Sugar", "Hbsag"]
                    if c in lab.columns]
        if lab_cols:
            fig, axes = plt.subplots(1, len(lab_cols), figsize=(16, 7))
            if len(lab_cols) == 1:
                axes = [axes]
            palette = sns.color_palette("Set1", len(lab_cols))
            for ax, col, color in zip(axes, lab_cols, palette):
                vc = lab[col].value_counts(dropna=False)
                sns.barplot(x=vc.index.astype(str), y=vc.values,
                            ax=ax, color=color, edgecolor="black")
                _style_ax(ax, col, ylabel="Count")
                plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
            _watermark(fig); plt.tight_layout()
            out.append(("Lab Test Results",
                        _save_docx(fig, "12_lab.png")))

    # ---- 13. Lab Turnaround Time ----
    if (not lab.empty
            and {"Created At", "Completed At"}.issubset(lab.columns)):
        tat = (pd.to_datetime(lab["Completed At"], errors="coerce") -
               pd.to_datetime(lab["Created At"], errors="coerce")).dt.total_seconds() / 3600
        tat = tat.dropna()
        if not tat.empty:
            fig, ax = plt.subplots(figsize=(16, 7))
            sns.histplot(tat, bins=25, kde=True, color="#457b9d",
                         ax=ax, edgecolor="white")
            _style_ax(ax, "Lab Turnaround Time",
                      xlabel="Hours", ylabel="Tests")
            _watermark(fig); plt.tight_layout()
            out.append(("Lab Turnaround Time",
                        _save_docx(fig, "13_lab_tat.png")))

    # ---- 14. Malaria Positive Rate by Age ----
    if (not lab.empty and "Malaria Parasite" in lab.columns
            and not p.empty and "Age Group" in p.columns):
        m = lab.merge(p[["Id", "Age Group"]], left_on="Patient Id",
                       right_on="Id", how="left")
        m["_pos"] = m["Malaria Parasite"].astype(str).str.lower().isin(
            ["positive", "pos", "reactive", "+", "1", "true"])
        grp = m.groupby("Age Group")["_pos"].mean().mul(100).round(1)
        if not grp.empty:
            fig, ax = plt.subplots(figsize=(14, 7))
            sns.barplot(x=grp.index.astype(str), y=grp.values,
                        palette=_safe_palette("Reds"), ax=ax, edgecolor="black")
            _style_ax(ax, "Malaria Positive Rate by Age Group",
                      xlabel="Age Group", ylabel="Positive Rate (%)")
            _watermark(fig); plt.tight_layout()
            out.append(("Malaria Rate by Age",
                        _save_docx(fig, "14_malaria_age.png")))

    # ---- 15. Optical Walk-in vs Referred ----
    if not optical.empty and "Is Walk In" in optical.columns:
        vc = optical["Is Walk In"].value_counts()
        if not vc.empty:
            fig, ax = plt.subplots(figsize=(12, 8))
            ax.pie(vc.values, labels=vc.index, autopct="%1.1f%%",
                    colors=sns.color_palette("Paired"),
                    startangle=90,
                    textprops={"fontsize": 14, "fontweight": "bold"})
            ax.set_title("Walk-in vs Referred (Optical)",
                         fontsize=20, fontweight="bold", color="#0b4a6f")
            _watermark(fig); plt.tight_layout()
            out.append(("Optical Walk-in vs Referred",
                        _save_docx(fig, "15_optical_walkin.png")))

    # ---- 16. Visual Acuity ----
    if not optical.empty:
        fig, axes = plt.subplots(1, 2, figsize=(16, 7))
        if "Visual Acuity Left" in optical.columns and optical["Visual Acuity Left"].notna().any():
            vc = optical["Visual Acuity Left"].value_counts()
            sns.barplot(x=vc.index.astype(str), y=vc.values,
                        palette=_safe_palette("Blues"), ax=axes[0], edgecolor="black")
            _style_ax(axes[0], "Visual Acuity – Left Eye", ylabel="Count")
            plt.setp(axes[0].get_xticklabels(), rotation=30, ha="right")
        else:
            axes[0].axis("off")
        if "Visual Acuity Right" in optical.columns and optical["Visual Acuity Right"].notna().any():
            vc = optical["Visual Acuity Right"].value_counts()
            sns.barplot(x=vc.index.astype(str), y=vc.values,
                        palette=_safe_palette("Greens"), ax=axes[1], edgecolor="black")
            _style_ax(axes[1], "Visual Acuity – Right Eye", ylabel="Count")
            plt.setp(axes[1].get_xticklabels(), rotation=30, ha="right")
        else:
            axes[1].axis("off")
        _watermark(fig); plt.tight_layout()
        out.append(("Visual Acuity",
                    _save_docx(fig, "16_acuity.png")))

    # ---- 17. Glasses Type ----
    if (not optical.empty and "Glasses Type" in optical.columns
            and optical["Glasses Type"].notna().any()):
        vc = optical["Glasses Type"].value_counts()
        if not vc.empty:
            fig, ax = plt.subplots(figsize=(14, 7))
            sns.barplot(x=vc.index.astype(str), y=vc.values,
                        palette=_safe_palette("pastel"), ax=ax, edgecolor="black")
            _style_ax(ax, "Glasses Types Allocated", ylabel="Count")
            plt.xticks(rotation=20)
            _watermark(fig); plt.tight_layout()
            out.append(("Glasses Type",
                        _save_docx(fig, "17_glasses.png")))

    # ---- 18. Refractive Error ----
    if (not optical.empty and "Refractive Error" in optical.columns
            and optical["Refractive Error"].notna().any()):
        vc = optical["Refractive Error"].value_counts()
        if not vc.empty:
            fig, ax = plt.subplots(figsize=(14, 7))
            sns.barplot(x=vc.index.astype(str), y=vc.values,
                        palette=_safe_palette("husl"), ax=ax, edgecolor="black")
            _style_ax(ax, "Refractive Errors", ylabel="Count")
            plt.xticks(rotation=20)
            _watermark(fig); plt.tight_layout()
            out.append(("Refractive Error",
                        _save_docx(fig, "18_refractive.png")))

    # ---- 19. Top 15 Dispensed Drugs ----
    if not pharm.empty and "Drug Name" in pharm.columns:
        names = filter_real_drugs(pharm["Drug Name"])
        top = names.value_counts().head(15)
        if not top.empty:
            fig, ax = plt.subplots(figsize=(16, 9))
            sns.barplot(y=top.index, x=top.values, palette=_safe_palette("YlOrRd"),
                        ax=ax, edgecolor="black")
            _style_ax(ax, "Top 15 Dispensed Drugs",
                      xlabel="Number of Dispensations", ylabel="Drug")
            _watermark(fig); plt.tight_layout()
            out.append(("Top 15 Dispensed Drugs",
                        _save_docx(fig, "19_drugs.png")))

    # ---- 20. Pharmacy Dispense Rate ----
    if not pharm.empty and "Dispensed" in pharm.columns:
        vc = pharm["Dispensed"].value_counts()
        if not vc.empty:
            fig, ax = plt.subplots(figsize=(12, 8))
            ax.pie(vc.values, labels=vc.index, autopct="%1.1f%%",
                    colors=sns.color_palette("Set3"),
                    startangle=90,
                    textprops={"fontsize": 14, "fontweight": "bold"})
            ax.set_title("Pharmacy Dispense Rate",
                         fontsize=20, fontweight="bold", color="#0b4a6f")
            _watermark(fig); plt.tight_layout()
            out.append(("Pharmacy Dispense Rate",
                        _save_docx(fig, "20_dispense.png")))

    # ---- 21. Drug Stock by Category ----
    if (not drugs.empty
            and {"Quantity", "Reorder Level", "Category"}.issubset(drugs.columns)):
        d = drugs.copy()
        d["Stock Status"] = np.where(d["Quantity"] <= d["Reorder Level"],
                                      "Reorder", "OK")
        fig, ax = plt.subplots(figsize=(16, 8))
        sns.barplot(data=d, x="Category", y="Quantity", hue="Stock Status",
                    palette={"OK": "#2a9d8f", "Reorder": "#e63946"},
                    ax=ax, edgecolor="black")
        _style_ax(ax, "Drug Stock by Category",
                  ylabel="Quantity on Hand", xlabel="Category")
        plt.xticks(rotation=20)
        _watermark(fig); plt.tight_layout()
        out.append(("Drug Stock by Category",
                    _save_docx(fig, "21_stock.png")))

    # ---- 22. Age Pyramid by Gender ----
    if (not p.empty and "Age Group" in p.columns and "Gender" in p.columns):
        pyr = p.groupby(["Age Group", "Gender"]).size().reset_index(name="Count")
        if not pyr.empty:
            fig, ax = plt.subplots(figsize=(16, 9))
            sns.barplot(data=pyr, x="Count", y="Age Group", hue="Gender",
                        palette=_safe_palette("Set2"), orient="h",
                        ax=ax, edgecolor="black")
            _style_ax(ax, "Age Pyramid by Gender",
                      xlabel="Patients", ylabel="Age Group")
            _watermark(fig); plt.tight_layout()
            out.append(("Age Pyramid by Gender",
                        _save_docx(fig, "22_age_pyramid.png")))

    return out


# ============================================================
# Summary
# ============================================================
def build_summary(data):
    s = {}
    p = data.get("patient", pd.DataFrame())
    c = data.get("consultations", pd.DataFrame())
    lab = data.get("lab_tests", pd.DataFrame())
    ph = data.get("pharmacy", pd.DataFrame())
    d = data.get("drugs", pd.DataFrame())

    if not p.empty:
        s["Total Patients"] = p["Id"].nunique() if "Id" in p.columns else len(p)
        if "Gender" in p.columns:
            s["Gender Split"] = ", ".join(
                f"{k}: {v}" for k, v in p["Gender"].value_counts().items())
        if "Age" in p.columns and p["Age"].notna().any():
            s["Average Age"] = f"{p['Age'].mean():.1f} years"
        if "Is Admitted" in p.columns:
            adm = p["Is Admitted"].astype(str).str.lower().isin(["true", "yes", "1"]).sum()
            s["Admitted Patients"] = int(adm)
    if not c.empty and "Diagnosis" in c.columns:
        top = c["Diagnosis"].value_counts().head(5)
        s["Top 5 Diagnoses"] = " | ".join(f"{k} ({v})" for k, v in top.items())
    if not lab.empty and "Malaria Parasite" in lab.columns:
        pos = lab["Malaria Parasite"].astype(str).str.lower().isin(
            ["positive", "pos", "reactive", "+", "1", "true"]).sum()
        s["Malaria Positive Tests"] = int(pos)
    if not ph.empty and "Drug Name" in ph.columns:
        names = filter_real_drugs(ph["Drug Name"])
        top = names.value_counts().head(5)
        if not top.empty:
            s["Top 5 Dispensed Drugs"] = " | ".join(f"{k} ({v})" for k, v in top.items())
    if not d.empty and {"Quantity", "Reorder Level"}.issubset(d.columns):
        low = (d["Quantity"] <= d["Reorder Level"]).sum()
        s["Drugs Needing Reorder"] = int(low)
    return s


# ============================================================
# HTML template
# ============================================================
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1.0"/>
<title>COREP Analysis Report – {{ brand_line }}</title>
<style>
  :root { --primary:#0b4a6f; --accent:#1a7fad; --bg:#f7f9fb; --card:#fff;
          --text:#223; --muted:#666; }
  * { box-sizing: border-box; }
  html,body { margin:0; padding:0;
    font-family:'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
    background:var(--bg); color:var(--text); line-height:1.55; }
  .container { max-width:1100px; margin:0 auto; padding:clamp(16px,3vw,40px); }
  header.report-header {
    background:linear-gradient(135deg,var(--primary) 0%,var(--accent) 100%);
    color:#fff; padding:clamp(20px,4vw,40px); border-radius:12px;
    margin-bottom:clamp(16px,3vw,32px); box-shadow:0 4px 14px rgba(0,0,0,0.08); }
  header.report-header h1 { margin:0 0 6px; font-size:clamp(1.4rem,4vw,2rem); }
  header.report-header .meta { opacity:0.85; font-size:clamp(0.8rem,2vw,0.95rem); }
  h2 { color:var(--primary); font-size:clamp(1.1rem,3vw,1.5rem);
       border-bottom:2px solid var(--primary); padding-bottom:6px;
       margin:clamp(24px,4vw,40px) 0 16px; }
  .card { background:var(--card); padding:clamp(14px,2.5vw,24px);
          border-radius:10px; box-shadow:0 1px 4px rgba(0,0,0,0.06);
          margin-bottom:16px; }
  table.summary { width:100%; border-collapse:collapse;
                  font-size:clamp(0.85rem,2vw,1rem); }
  table.summary th, table.summary td {
    padding:10px 12px; border-bottom:1px solid #eee;
    text-align:left; vertical-align:top; }
  table.summary th { background:var(--primary); color:#fff;
                     font-weight:500; width:35%; }
  .chart-grid { display:grid;
    grid-template-columns:repeat(auto-fit,minmax(320px,1fr));
    gap:clamp(12px,2vw,20px); }
  .chart-card { background:var(--card); padding:clamp(10px,2vw,16px);
    border-radius:10px; box-shadow:0 1px 4px rgba(0,0,0,0.06);
    text-align:center; overflow:hidden; }
  .chart-card img { width:100%; height:auto; display:block; border-radius:6px; }
  .chart-card .caption { color:var(--muted);
    font-size:clamp(0.75rem,1.8vw,0.9rem); margin-top:8px; font-weight:500; }
  ul.insights { padding-left:20px; font-size:clamp(0.9rem,2vw,1rem); }
  ul.insights li { margin-bottom:8px; }
  footer.report-footer { margin-top:40px; text-align:center;
    color:var(--muted); font-size:0.9rem; line-height:1.6; }
  @media print {
    body { background:#fff; }
    .container { max-width:100%; padding:0; }
    header.report-header { box-shadow:none; border-radius:0; }
    .chart-grid { grid-template-columns:repeat(2,1fr); }
    .chart-card { page-break-inside:avoid; box-shadow:none;
                  border:1px solid #eee; }
    h2 { page-break-after:avoid; } }
  @media (max-width:480px) {
    table.summary th { width:45%; }
    .chart-grid { grid-template-columns:1fr; } }
</style>
</head>
<body>
<div class="container">
  <header class="report-header">
    <h1>🏥 COREP Clinical Data Analysis Report {{ trademark }}</h1>
    <div class="meta">Generated on {{ generated_at }}</div>
    <div class="meta" style="margin-top:6px; font-size:0.85em;">
      {{ brand_line }} &nbsp;·&nbsp; {{ developer_line }}
    </div>
  </header>

  <h2>1. Executive Summary</h2>
  <div class="card">
    <table class="summary">
      {% for k, v in summary.items() %}
      <tr><th>{{ k }}</th><td>{{ v }}</td></tr>
      {% endfor %}
    </table>
  </div>

  <h2>2. Charts &amp; Findings</h2>
  <div class="chart-grid">
    {% for name, b64 in charts %}
    <div class="chart-card">
      <img src="data:image/png;base64,{{ b64 }}" alt="{{ name }}"/>
      <div class="caption">{{ name.replace('_',' ').replace('.png','').title() }}</div>
    </div>
    {% endfor %}
  </div>

  <h2>3. Key Insights</h2>
  <div class="card">
    <ul class="insights">
      <li>Patient registrations show a consistent trend — monitor monthly volume.</li>
      <li>Top diagnoses drive most of the clinical workload.</li>
      <li>Referral pathway shows the strongest downstream load.</li>
      <li>Low-stock alerts must be actioned promptly.</li>
      <li>Use forecasting to plan drug &amp; staff demand.</li>
      <li>Patient clusters reveal distinct risk groups.</li>
    </ul>
  </div>

  <footer class="report-footer">
    <strong>{{ brand_line }}</strong><br>
    {{ powered_by }}<br>
    <em>{{ developer_line }}</em><br>
    <span style="font-size:0.8em;">{{ copyright }}</span>
  </footer>
</div>
</body>
</html>
"""


def generate_html_report(summary, charts):
    html = Template(HTML_TEMPLATE).render(
        generated_at=datetime.now().strftime("%Y-%m-%d %H:%M"),
        summary=summary, charts=charts,
        brand_line=BRAND_LINE, developer_line=DEVELOPER_LINE,
        copyright=COPYRIGHT, powered_by=POWERED_BY, trademark=TRADEMARK,
    )
    out = os.path.join(REPORT_DIR, "corep_report2.html")
    with open(out, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"📄 Responsive HTML report -> {out}")
    return out, html


# ============================================================
# WORD report
# ============================================================
def generate_word_report(summary, chart_items, data=None):
    try:
        from docx import Document
        from docx.shared import Inches, Pt, RGBColor
        from docx.enum.text import WD_ALIGN_PARAGRAPH
        from docx.enum.section import WD_ORIENT
    except ImportError:
        print("⚠️  python-docx not installed. Run: pip install python-docx")
        return None

    doc = Document()

    for section in doc.sections:
        new_w, new_h = section.page_height, section.page_width
        section.orientation = WD_ORIENT.LANDSCAPE
        section.page_width  = new_w
        section.page_height = new_h
        section.top_margin    = Inches(0.5)
        section.bottom_margin = Inches(0.5)
        section.left_margin   = Inches(0.5)
        section.right_margin  = Inches(0.5)

    IMG_WIDTH = Inches(9.7)

    title = doc.add_heading(f"COREP Clinical Data Analysis Report {TRADEMARK}", level=0)
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    for r in title.runs:
        r.font.color.rgb = RGBColor(11, 74, 111)

    sub = doc.add_paragraph(
        f"Generated on {datetime.now().strftime('%Y-%m-%d %H:%M')}   |   "
        f"Landscape · 0.5\" margins"
    )
    sub.alignment = WD_ALIGN_PARAGRAPH.CENTER
    sub.runs[0].font.size = Pt(10)
    sub.runs[0].font.color.rgb = RGBColor(120, 120, 120)

    brand_p = doc.add_paragraph()
    brand_p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = brand_p.add_run(BRAND_LINE)
    r.font.size = Pt(14); r.font.bold = True
    r.font.color.rgb = RGBColor(11, 74, 111)

    dev_p = doc.add_paragraph()
    dev_p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = dev_p.add_run(DEVELOPER_LINE_FULL)
    r.font.size = Pt(11); r.font.italic = True
    r.font.color.rgb = RGBColor(80, 80, 80)

    copy_p = doc.add_paragraph()
    copy_p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = copy_p.add_run(COPYRIGHT)
    r.font.size = Pt(9); r.font.color.rgb = RGBColor(130, 130, 130)

    doc.add_paragraph()

    doc.add_heading("1. Executive Summary", level=1)
    table = doc.add_table(rows=0, cols=2)
    table.style = "Light Grid Accent 1"
    for k, v in summary.items():
        row = table.add_row().cells
        row[0].text = str(k); row[1].text = str(v)
        for para in row[0].paragraphs:
            for r in para.runs:
                r.bold = True; r.font.size = Pt(12)
        for para in row[1].paragraphs:
            for r in para.runs:
                r.font.size = Pt(12)

    doc.add_page_break()

    doc.add_heading("2. Charts & Findings", level=1)
    for i, (caption, path) in enumerate(chart_items, start=1):
        h = doc.add_heading(f"2.{i} {caption}", level=2)
        for r in h.runs:
            r.font.color.rgb = RGBColor(11, 74, 111)
        p = doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        try:
            run = p.add_run(); run.add_picture(path, width=IMG_WIDTH)
        except Exception as e:
            doc.add_paragraph(f"[Chart could not be embedded: {e}]")
        if i < len(chart_items):
            doc.add_page_break()

    if data is not None and not data.get("consultations", pd.DataFrame()).empty:
        cons = data["consultations"]
        if "Treatment Plan (Clinical)" in cons.columns:
            doc.add_page_break()
            doc.add_heading("3. Treatment Plans (Clinical Only)", level=1)
            note = doc.add_paragraph(
                "Prescription-related text has been removed. "
                "Only clinical/non-pharmacy instructions are shown."
            )
            note.runs[0].font.size = Pt(9)
            note.runs[0].font.color.rgb = RGBColor(120, 120, 120)

            t = doc.add_table(rows=1, cols=2)
            t.style = "Light Grid Accent 1"
            hdr = t.rows[0].cells
            hdr[0].text = "Consultation ID"
            hdr[1].text = "Treatment Plan (Clinical)"
            for c in hdr:
                for para in c.paragraphs:
                    for r in para.runs:
                        r.bold = True

            sub = cons[cons["Treatment Plan (Clinical)"].astype(str).str.strip().ne("")]
            for _, row in sub.head(30).iterrows():
                cells = t.add_row().cells
                cells[0].text = str(row.get("Id", ""))
                cells[1].text = str(row["Treatment Plan (Clinical)"])

    doc.add_page_break()
    doc.add_heading("4. Key Insights", level=1)
    for item in [
        "Patient registrations show a consistent trend — monitor monthly volume for resource planning.",
        "Top diagnoses drive most of the clinical workload — target training and drug stocking accordingly.",
        "Referral pathway indicates the strongest downstream load (Lab vs Pharmacy vs Optical).",
        "Low-stock alerts must be actioned promptly to avoid stockouts.",
        "Use the forecasting module to plan drug & staff demand for the coming month.",
        "Patient clusters reveal distinct risk groups for targeted interventions.",
    ]:
        doc.add_paragraph(item, style="List Bullet")

    footer_text = f"{BRAND_LINE} · {DEVELOPER_LINE_SHORT} · {COPYRIGHT}"
    for section in doc.sections:
        footer = section.footer
        para = footer.paragraphs[0] if footer.paragraphs else footer.add_paragraph()
        para.alignment = WD_ALIGN_PARAGRAPH.CENTER
        para.text = footer_text
        for r in para.runs:
            r.font.size = Pt(8)
            r.font.color.rgb = RGBColor(130, 130, 130)

    out = os.path.join(REPORT_DIR, "corep_report2.docx")
    doc.save(out)
    print(f"📝 Word report (landscape, 0.5\" margins, branded) -> {out}")
    return out


# ============================================================
# PDF (optional)
# ============================================================
def generate_pdf_report(html_str):
    try:
        from weasyprint import HTML
    except Exception as e:
        print(f"ℹ️  PDF skipped (WeasyPrint unavailable): {e}")
        return None
    try:
        out = os.path.join(REPORT_DIR, "corep_report.pdf")
        HTML(string=html_str, base_url=os.getcwd()).write_pdf(
            out, presentational_hints=True)
        print(f"📕 PDF report -> {out}")
        return out
    except Exception as e:
        print(f"ℹ️  PDF skipped (WeasyPrint failed): {e}")
        print("    → Fix: install GTK3 runtime or use conda-forge.")
        return None


# ============================================================
# Main
# ============================================================
def generate_report(data=None):
    print("\n=== GENERATING REPORT ===")
    if data is None:
        import main_analysis as analysis
        data = analysis.clean_all(analysis.load_data())

    if not data.get("consultations", pd.DataFrame()).empty:
        data["consultations"] = clean_treatment_plans(data["consultations"])
        print("✅ Treatment plans cleaned (prescriptions removed).")

    summary = build_summary(data)

    chart_files = collect_charts()
    charts_b64 = [(name, img_to_base64(path)) for name, path in chart_files]

    print("🎨 Rendering large colorful charts for Word report ...")
    docx_charts = render_docx_charts(data)
    print(f"   -> {len(docx_charts)} charts rendered")

    html_path, html_str = generate_html_report(summary, charts_b64)
    word_path = generate_word_report(summary, docx_charts, data=data)
    pdf_path  = generate_pdf_report(html_str)

    print("\n📦 Report outputs:")
    print(f"   • HTML: {html_path}")
    if word_path: print(f"   • Word: {word_path}")
    if pdf_path:  print(f"   • PDF : {pdf_path}")

    return {"html": html_path, "word": word_path, "pdf": pdf_path}


if __name__ == "__main__":
    generate_report()