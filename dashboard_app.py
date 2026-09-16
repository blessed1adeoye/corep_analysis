# Interactive Streamlit Dashboard

"""
Streamlit dashboard for COREP data.
Run:  streamlit run dashboard_app.py
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="COREP Dashboard", layout="wide", page_icon="🏥")

EXCEL_FILE = "corep_data.xlsx"


@st.cache_data
def load():
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
    for k, s in sheets.items():
        try:
            data[k] = pd.read_excel(EXCEL_FILE, sheet_name=s)
        except Exception:
            data[k] = pd.DataFrame()
    # enrich patient
    p = data["patient"]
    if "Date Of Birth" in p.columns:
        p["Date Of Birth"] = pd.to_datetime(p["Date Of Birth"], errors="coerce")
        p["Age"] = ((pd.Timestamp.today() - p["Date Of Birth"]).dt.days / 365.25).round(1)
        p["Age Group"] = pd.cut(p["Age"], [0, 5, 12, 18, 35, 50, 65, 120],
                                 labels=["0-5", "6-12", "13-18", "19-35", "36-50", "51-65", "65+"])
    data["patient"] = p
    return data


data = load()

st.title("🏥 COREP Clinical Data Dashboard")
st.caption("Interactive analysis of patient, clinical, lab, optical and pharmacy data.")

# ---- Sidebar filters ----
st.sidebar.header("Filters")
patient = data["patient"]

age_range = st.sidebar.slider("Age range",
    int(patient["Age"].min()) if "Age" in patient else 0,
    int(patient["Age"].max()) if "Age" in patient else 100,
    (0, 100))

genders = st.sidebar.multiselect("Gender",
    options=sorted(patient["Gender"].dropna().unique()) if "Gender" in patient else [],
    default=list(patient["Gender"].dropna().unique()) if "Gender" in patient else [])

flt = patient.copy()
if "Age" in flt: flt = flt[(flt["Age"] >= age_range[0]) & (flt["Age"] <= age_range[1])]
if genders and "Gender" in flt: flt = flt[flt["Gender"].isin(genders)]

patient_ids = set(flt["Id"])

# ---- KPI row ----
c1, c2, c3, c4 = st.columns(4)
c1.metric("Patients", f"{len(flt):,}")
c2.metric("Consultations", f"{data['consultations'].query('`Patient Id` in @patient_ids').shape[0]:,}"
          if not data["consultations"].empty else 0)
c3.metric("Lab Tests", f"{data['lab_tests'].query('`Patient Id` in @patient_ids').shape[0]:,}"
          if not data["lab_tests"].empty else 0)
c4.metric("Pharmacy Orders", f"{data['pharmacy'].query('`Patient Id` in @patient_ids').shape[0]:,}"
          if not data["pharmacy"].empty else 0)

# ---- Tabs ----
tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["👥 Demographics", "🩺 Consultations", "🧪 Lab", "👁 Optical", "💊 Pharmacy"])

with tab1:
    col1, col2 = st.columns(2)
    if "Gender" in flt:
        col1.plotly_chart(px.pie(flt, names="Gender", title="Gender Split",
                                  hole=0.4), use_container_width=True)
    if "Age Group" in flt:
        col2.plotly_chart(px.histogram(flt, x="Age Group", title="Age Group"),
                          use_container_width=True)
    if "Created At" in flt:
        flt2 = flt.copy()
        flt2["Created At"] = pd.to_datetime(flt2["Created At"], errors="coerce")
        flt2["Month"] = flt2["Created At"].dt.to_period("M").astype(str)
        trend = flt2.groupby("Month").size().reset_index(name="Registrations")
        st.plotly_chart(px.line(trend, x="Month", y="Registrations",
                                 title="Monthly Registrations", markers=True),
                        use_container_width=True)

with tab2:
    cons = data["consultations"]
    if not cons.empty and "Patient Id" in cons:
        cons = cons[cons["Patient Id"].isin(patient_ids)]
        if "Diagnosis" in cons:
            top = cons["Diagnosis"].value_counts().head(15).reset_index()
            top.columns = ["Diagnosis", "Count"]
            st.plotly_chart(px.bar(top, x="Count", y="Diagnosis", orientation="h",
                                    title="Top 15 Diagnoses"), use_container_width=True)

with tab3:
    lab = data["lab_tests"]
    if not lab.empty and "Patient Id" in lab:
        lab = lab[lab["Patient Id"].isin(patient_ids)]
        for col in ["Malaria Parasite", "Random Blood Sugar", "Hbsag"]:
            if col in lab.columns:
                vc = lab[col].value_counts(dropna=False).reset_index()
                vc.columns = ["Result", "Count"]
                st.plotly_chart(px.bar(vc, x="Result", y="Count", title=col),
                                use_container_width=True)

with tab4:
    opt = data["optical"]
    if not opt.empty and "Patient Id" in opt:
        opt = opt[opt["Patient Id"].isin(patient_ids)]
        if "Is Walk In" in opt:
            vc = opt["Is Walk In"].value_counts().reset_index()
            vc.columns = ["Walk In", "Count"]
            st.plotly_chart(px.pie(vc, names="Walk In", values="Count",
                                    title="Walk-in vs Referred"), use_container_width=True)

with tab5:
    ph = data["pharmacy"]
    if not ph.empty and "Patient Id" in ph:
        ph = ph[ph["Patient Id"].isin(patient_ids)]
        if "Drug Name" in ph:
            top = ph["Drug Name"].value_counts().head(15).reset_index()
            top.columns = ["Drug", "Count"]
            st.plotly_chart(px.bar(top, x="Count", y="Drug", orientation="h",
                                    title="Top 15 Drugs"), use_container_width=True)
        if "Dispensed" in ph:
            vc = ph["Dispensed"].value_counts().reset_index()
            vc.columns = ["Status", "Count"]
            st.plotly_chart(px.pie(vc, names="Status", values="Count",
                                    title="Dispensed Rate"), use_container_width=True)

st.markdown("---")
st.caption("Built with Streamlit + Plotly | COREP Analytics")