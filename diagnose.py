"""Diagnose why ML models skip and dashboard shows 0."""
import pandas as pd

xls = pd.ExcelFile("corep_data.xlsx")

print("=" * 70)
print("1. PATIENTS SHEET")
print("=" * 70)
p = pd.read_excel(xls, sheet_name="Patients")
print(f"Shape: {p.shape}")
print(f"Columns: {list(p.columns)}")
print(f"\n'Id' dtype: {p['Id'].dtype}")
print(f"'Id' sample: {p['Id'].head(3).tolist()}")
print(f"\n'Is Admitted' value_counts:")
print(p["Is Admitted"].value_counts(dropna=False))
print(f"dtype: {p['Is Admitted'].dtype}")

print("\n" + "=" * 70)
print("2. CONSULTATIONS SHEET")
print("=" * 70)
c = pd.read_excel(xls, sheet_name="Consultations")
print(f"Shape: {c.shape}")
print(f"Columns: {list(c.columns)}")
if "Patient Id" in c.columns:
    print(f"\n'Patient Id' dtype: {c['Patient Id'].dtype}")
    print(f"'Patient Id' sample: {c['Patient Id'].head(3).tolist()}")
    print(f"Unique patients in consultations: {c['Patient Id'].nunique()}")
    print(f"Overlap with Patients.Id: {len(set(c['Patient Id']) & set(p['Id']))}")

print("\n" + "=" * 70)
print("3. LAB TESTS SHEET")
print("=" * 70)
l = pd.read_excel(xls, sheet_name="Lab_Tests")
print(f"Shape: {l.shape}")
print(f"Columns: {list(l.columns)}")
if "Malaria Parasite" in l.columns:
    print(f"\n'Malaria Parasite' value_counts:")
    print(l["Malaria Parasite"].value_counts(dropna=False))
    print(f"dtype: {l['Malaria Parasite'].dtype}")
if "Patient Id" in l.columns:
    print(f"\n'Patient Id' dtype: {l['Patient Id'].dtype}")
    print(f"'Patient Id' sample: {l['Patient Id'].head(3).tolist()}")

print("\n" + "=" * 70)
print("4. PHARMACY ORDERS SHEET")
print("=" * 70)
ph = pd.read_excel(xls, sheet_name="Pharmacy_Orders")
print(f"Shape: {ph.shape}")
print(f"Columns: {list(ph.columns)}")
if "Patient Id" in ph.columns:
    print(f"\n'Patient Id' dtype: {ph['Patient Id'].dtype}")
    print(f"'Patient Id' sample: {ph['Patient Id'].head(3).tolist()}")

print("\n" + "=" * 70)
print("5. DTYPE OVERLAP CHECK")
print("=" * 70)
print(f"Patients.Id dtype        : {p['Id'].dtype}")
if "Patient Id" in c.columns:
    print(f"Consultations.Patient Id : {c['Patient Id'].dtype}")
    print(f"  → Same? {p['Id'].dtype == c['Patient Id'].dtype}")
if "Patient Id" in l.columns:
    print(f"Lab_Tests.Patient Id     : {l['Patient Id'].dtype}")
    print(f"  → Same? {p['Id'].dtype == l['Patient Id'].dtype}")