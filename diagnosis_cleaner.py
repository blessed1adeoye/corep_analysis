"""
Clinical diagnosis text normalization.
- Splits multi-diagnosis cells into individual diagnoses
- Maps synonyms and abbreviations to canonical forms
- Rebuilds the Diagnosis column for clean analytics
"""

import re
import pandas as pd


# ============================================================
# STEP 1 — Canonical diagnosis mapping
# ============================================================
# Keys = variations seen in the data (lowercase)
# Values = canonical (standard) form
DIAGNOSIS_MAP = {
    # --- Hypertension ---
    "htn": "Hypertension",
    "hbp": "Hypertension",
    "high blood pressure": "Hypertension",
    "elevated bp": "Hypertension",
    "elevated blood pressure": "Hypertension",
    "hypertention": "Hypertension",   # common typo
    "hypertension": "Hypertension",

    # --- Diabetes ---
    "dm": "Diabetes Mellitus",
    "diabetes": "Diabetes Mellitus",
    "diabetes mellitus": "Diabetes Mellitus",
    "t2dm": "Type 2 Diabetes Mellitus",
    "type 2 diabetes": "Type 2 Diabetes Mellitus",
    "type 2 dm": "Type 2 Diabetes Mellitus",
    "tidm": "Type 1 Diabetes Mellitus",

    # --- Malaria ---
    "malaria": "Malaria",
    "mp": "Malaria",
    "malaria parasite": "Malaria",
    "positive malaria": "Malaria",
    "malaria positive": "Malaria",

    # --- Typhoid ---
    "typhoid": "Typhoid Fever",
    "typhoid fever": "Typhoid Fever",
    "enteric fever": "Typhoid Fever",

    # --- Peptic Ulcer Disease ---
    "pud": "Peptic Ulcer Disease",
    "peptic ulcer": "Peptic Ulcer Disease",
    "peptic ulcer disease": "Peptic Ulcer Disease",
    "gastric ulcer": "Peptic Ulcer Disease",
    "duodenal ulcer": "Peptic Ulcer Disease",

    # --- Arthritis ---
    "arthritis": "Arthritis",
    "oa": "Osteoarthritis",
    "osteoarthritis": "Osteoarthritis",
    "ra": "Rheumatoid Arthritis",
    "rheumatoid arthritis": "Rheumatoid Arthritis",
    "polyarthritis": "Polyarthritis",

    # --- Respiratory ---
    "cough": "Cough",
    "pneumonia": "Pneumonia",
    "asthma": "Asthma",
    "uri": "Upper Respiratory Infection",
    "upper respiratory infection": "Upper Respiratory Infection",

    # --- GI ---
    "gastritis": "Gastritis",
    "gerd": "GERD",
    "acid reflux": "GERD",

    # --- Others ---
    "fever": "Fever",
    "headache": "Headache",
    "anxiety": "Anxiety",
    "allergies": "Allergies",
    "allergy": "Allergies",
    "refractive error": "Refractive Error",
    "hypertension": "Hypertension",
    "stable": None,       # ← non-diagnosis filler
    "healthy": None,
    "well": None,
}


# ============================================================
# STEP 2 — Splitting
# ============================================================
# Characters that separate multiple diagnoses in one cell
SPLIT_PATTERN = re.compile(r"[,;/\|]|\band\b|\+", flags=re.IGNORECASE)


def split_diagnoses(text):
    """Split a multi-diagnosis string into individual diagnoses."""
    if pd.isna(text) or str(text).strip() == "":
        return []
    parts = SPLIT_PATTERN.split(str(text))
    return [p.strip() for p in parts if p.strip()]


# ============================================================
# STEP 3 — Normalization
# ============================================================
def normalize_diagnosis(text):
    """Map a single diagnosis to its canonical form."""
    if pd.isna(text) or str(text).strip() == "":
        return None
    key = str(text).strip().lower()

    # Direct lookup
    if key in DIAGNOSIS_MAP:
        return DIAGNOSIS_MAP[key]  # ← can be None (for fillers)

    # Partial match (e.g., "hypertension stage 2" → "Hypertension")
    for k, v in DIAGNOSIS_MAP.items():
        if k in key:
            return v

    # Fallback: title-case the original
    return str(text).strip().title()


def normalize_multi_diagnoses(text):
    """Split + normalize + dedupe + rejoin."""
    parts = split_diagnoses(text)
    normalized = []
    for p in parts:
        n = normalize_diagnosis(p)
        if n and n not in normalized:      # skip None and duplicates
            normalized.append(n)
    return " | ".join(normalized) if normalized else None


# ============================================================
# STEP 4 — Apply to a DataFrame
# ============================================================
def clean_diagnosis_column(df, column="Diagnosis"):
    """
    Returns a new DataFrame with a normalized Diagnosis column
    plus a long-format DataFrame for counting.
    """
    if df is None or df.empty or column not in df.columns:
        return df, pd.DataFrame()

    df = df.copy()
    df["Diagnosis_Clean"] = df[column].apply(normalize_multi_diagnoses)

    # Long format — one row per individual diagnosis
    long_rows = []
    for idx, row in df.iterrows():
        diag_str = row.get("Diagnosis_Clean")
        if pd.isna(diag_str) or not diag_str:
            continue
        for d in str(diag_str).split(" | "):
            long_rows.append({
                "Consultation Id": row.get("Id", idx),
                "Patient Id": row.get("Patient Id"),
                "Diagnosis": d.strip(),
            })

    long_df = pd.DataFrame(long_rows)
    return df, long_df


# ============================================================
# CLI test
# ============================================================
if __name__ == "__main__":
    # Quick test cases
    test_cases = [
        "HTN",
        "Hypertension",
        "htn pud",
        "HTN, PUD",
        "Malaria / Typhoid",
        "HTN and DM",
        "Malaria Parasite",
        "mp",
        "stable",
        "hypertension stage 2",
        "",
        None,
    ]
    print("Testing normalizer:")
    for t in test_cases:
        result = normalize_multi_diagnoses(t)
        print(f"  {repr(t):35s} → {repr(result)}")