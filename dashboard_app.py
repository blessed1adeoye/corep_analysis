"""
Streamlit dashboard for COREP data — EXPANDED, branded, PII-safe.
Optimized for single-day medical outreach data.
Run:  streamlit run dashboard_app.py
"""

import os
import time
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from main_analysis import filter_real_drugs, _match_sheet, EXCEL_FILE
from branding import (
    BRAND_LINE, BRAND_LINE_FULL,
    DEVELOPER_LINE, DEVELOPER_LINE_FULL, DEVELOPER_LINE_SHORT,
    COPYRIGHT, POWERED_BY, APP_TITLE, TRADEMARK,
    LOGO_PATH, DEVELOPER_NAME, COMPANY_NAME,
)
import httpx
import ollama


st.set_page_config(
    page_title=APP_TITLE,
    layout="wide",
    page_icon="🏥",
    initial_sidebar_state="expanded",
)

# Brand logo — renders in the top-left and sidebar
try:
    st.logo(
        "assets/logo.png",
        size="large",
        link="https://github.com/blessed1adeoye/corep_analysis",
    )
except Exception as e:
    st.sidebar.warning(f"Logo not found: {e}")


# ============================================================
# API Configuration
# ============================================================
def _get_api_url():
    try:
        return st.secrets.get("API_URL", "http://localhost:8000")
    except Exception:
        return "http://localhost:8000"


API_URL = _get_api_url()


def call_api(endpoint: str, payload: dict, timeout: float = 8.0) -> dict | None:
    """Call the COREP ML API. Returns dict or None on failure."""
    try:
        r = httpx.post(f"{API_URL}{endpoint}", json=payload, timeout=timeout)
        r.raise_for_status()
        return r.json()
    except httpx.ConnectError:
        return {"error": f"Cannot reach API at {API_URL}. Is it running?"}
    except httpx.TimeoutException:
        return {"error": "API request timed out."}
    except httpx.HTTPStatusError as e:
        return {"error": f"API error {e.response.status_code}: {e.response.text[:200]}"}
    except Exception as e:
        return {"error": f"Unexpected error: {e}"}


# ============================================================
# Styling
# ============================================================
st.markdown("""
<style>
    .main .block-container { padding-top: 2rem; padding-bottom: 2rem; }
    h1, h2, h3 { color: #0b4a6f; }
    div[data-testid="stMetricValue"] { font-size: 1.8rem; color: #0b4a6f; }
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] {
        height: 42px; padding: 0 14px; font-weight: 600;
        background: #eef4f9; border-radius: 8px 8px 0 0;
    }
    .outreach-banner {
        background: linear-gradient(135deg, #0b4a6f 0%, #1a7fad 100%);
        color: #fff; padding: 16px 24px; border-radius: 12px;
        margin-bottom: 20px; box-shadow: 0 4px 12px rgba(0,0,0,0.08);
    }
    .outreach-banner h2 { color: #fff !important; margin: 0 0 6px; }
    .outreach-banner p { margin: 0; opacity: 0.9; font-size: 0.95rem; }
</style>
""", unsafe_allow_html=True)


# ============================================================
# Global PII filter
# ============================================================
PII_COLUMNS = {
    "First Name", "Last Name", "Middle Name",
    "Phone", "Email", "Address",
    "Date Of Birth", "Hospital Number",
    "Created By Id", "Updated By Id",
    "Completed By Id", "Dispensed By Id",
}


def hide_pii(df):
    if df is None or df.empty:
        return df
    return df.drop(columns=[c for c in df.columns if c in PII_COLUMNS],
                    errors="ignore")


# ============================================================
# Load
# ============================================================
@st.cache_data
def load():
    xls = pd.ExcelFile(EXCEL_FILE)
    available = xls.sheet_names

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
        try:
            data[key] = pd.read_excel(xls, sheet_name=matched) if matched else pd.DataFrame()
        except Exception:
            data[key] = pd.DataFrame()

    # Derive Age
    p = data["patient"]
    if not p.empty and "Date Of Birth" in p.columns:
        p["Date Of Birth"] = pd.to_datetime(p["Date Of Birth"], errors="coerce")
        p["Age"] = ((pd.Timestamp.today() - p["Date Of Birth"]).dt.days / 365.25).round(1)
        p["Age Group"] = pd.cut(p["Age"], [0, 5, 12, 18, 35, 50, 65, 120],
                                 labels=["0-5", "6-12", "13-18", "19-35",
                                         "36-50", "51-65", "65+"])
    data["patient"] = p

    def _norm(df):
        if "Patient Id" in df.columns:
            df["Patient Id"] = pd.to_numeric(df["Patient Id"], errors="coerce").astype("Int64")
        if "Consultation Id" in df.columns:
            df["Consultation Id"] = pd.to_numeric(df["Consultation Id"], errors="coerce").astype("Int64")
        return df

    if not data["patient"].empty and "Id" in data["patient"].columns:
        data["patient"]["Id"] = pd.to_numeric(data["patient"]["Id"], errors="coerce").astype("Int64")
    for k in ["consultations", "lab_tests", "optical", "pharmacy", "nursing"]:
        data[k] = _norm(data[k])

    return data


data = load()


# ============================================================
# Outreach banner
# ============================================================
def _outreach_date(patient):
    if patient.empty or "Created At" not in patient.columns:
        return "an outreach event"
    dt = pd.to_datetime(patient["Created At"], errors="coerce").dropna()
    if dt.empty:
        return "an outreach event"
    d = dt.dt.date.value_counts().idxmax()
    return d.strftime("%d %B %Y")


st.markdown(f"""
<div class="outreach-banner">
    <h2>🏥 COREP Medical Outreach Dashboard {TRADEMARK}</h2>
    <p>Single-day clinical outreach · {_outreach_date(data['patient'])} ·
    Interactive analysis of patients, consultations, labs, optical, pharmacy &amp; vitals.</p>
    <p style="margin-top:8px; font-size:0.85rem; opacity:0.9;">
        {DEVELOPER_LINE_FULL}
    </p>
</div>
""", unsafe_allow_html=True)


# ============================================================
# Diagnostics
# ============================================================
with st.expander("🔬 Data diagnostics", expanded=False):
    c1, c2 = st.columns(2)
    with c1:
        st.write("**Rows per sheet**")
        for k, v in data.items():
            st.write(f"- `{k}`: {len(v)} rows × {len(v.columns)} cols")
    with c2:
        st.write("**ID overlap with Patients**")
        pids = set(data["patient"]["Id"].dropna()) if not data["patient"].empty else set()
        for k in ["consultations", "lab_tests", "pharmacy", "optical", "nursing"]:
            df = data[k]
            if "Patient Id" in df.columns:
                ov = len(set(df["Patient Id"].dropna()) & pids)
                st.write(f"- `{k}`: {ov} / {df['Patient Id'].nunique()} match")


# ============================================================
# Sidebar Filters
# ============================================================
if LOGO_PATH and os.path.exists(LOGO_PATH):
    st.sidebar.image(LOGO_PATH, use_column_width=True)

st.sidebar.header("🎛️ Filters")
if st.sidebar.button("🔄 Reset all filters"):
    st.cache_data.clear()
    st.rerun()

patient = data["patient"]

if not patient.empty and "Age" in patient.columns and patient["Age"].notna().any():
    amn = int(patient["Age"].min(skipna=True))
    amx = int(patient["Age"].max(skipna=True))
    if amn == amx:
        age_range = (amn, amx)
        st.sidebar.caption(f"Age: {amn} (all patients)")
    else:
        age_range = st.sidebar.slider("Age range", amn, amx, (amn, amx))
else:
    age_range = (0, 100)

ug = sorted(patient["Gender"].dropna().astype(str).unique()) \
     if not patient.empty and "Gender" in patient.columns else []
genders = st.sidebar.multiselect("Gender", options=ug, default=ug)

if not patient.empty and "Is Admitted" in patient.columns:
    adm_vals = sorted(patient["Is Admitted"].dropna().astype(str).unique())
else:
    adm_vals = []
adm_sel = st.sidebar.multiselect("Admission status", options=adm_vals, default=adm_vals)

flt = patient.copy()
if not flt.empty and "Age" in flt.columns and flt["Age"].notna().any():
    flt = flt[(flt["Age"] >= age_range[0]) & (flt["Age"] <= age_range[1])]
if genders and "Gender" in flt.columns:
    flt = flt[flt["Gender"].astype(str).isin(genders)]
if adm_sel and "Is Admitted" in flt.columns:
    flt = flt[flt["Is Admitted"].astype(str).isin(adm_sel)]

patient_ids = set(flt["Id"].dropna()) if "Id" in flt.columns else set()


def filt(df):
    if df.empty or "Patient Id" not in df.columns:
        return df.iloc[0:0]
    return df[df["Patient Id"].isin(patient_ids)]


cons = filt(data["consultations"])
lab = filt(data["lab_tests"])
opt = filt(data["optical"])
ph = filt(data["pharmacy"])
nurse = filt(data["nursing"])
drugs = data["drugs"]

# Sidebar branding
st.sidebar.markdown("---")
st.sidebar.markdown(
    f"""
    <div style="text-align:center; color:#666; font-size:0.8rem; line-height:1.5;">
        <strong style="color:#0b4a6f;">{BRAND_LINE}</strong><br>
        <em style="font-size:0.75rem;">{DEVELOPER_LINE_SHORT}</em><br>
        <span style="font-size:0.7rem; opacity:0.8;">{COPYRIGHT}</span>
    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# KPI Row
# ============================================================
st.markdown("### 📊 Key Metrics")
k1, k2, k3, k4, k5, k6 = st.columns(6)
k1.metric("👥 Patients", f"{len(flt):,}")
k2.metric("🩺 Consultations", f"{len(cons):,}")
k3.metric("🧪 Lab Tests", f"{len(lab):,}")
k4.metric("👁 Optical", f"{len(opt):,}")
k5.metric("💊 Pharmacy", f"{len(ph):,}")
k6.metric("🩺 Vitals", f"{len(nurse):,}")


# ============================================================
# Tabs
# ============================================================
tabs = st.tabs([
    "👥 Demographics",
    "🩺 Consultations",
    "🧪 Lab Tests",
    "👁 Optical",
    "💊 Pharmacy",
    "🩺 Vitals",
    "📈 Trends",
    "📋 Tables",
    "🔮 Predict",
    "🤖 Ask BGIE AI",
])
tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9, tab10 = tabs


# ------------------------------------------------------------
# TAB 1 — DEMOGRAPHICS
# ------------------------------------------------------------
with tab1:
    if flt.empty:
        st.info("No patient data in current filter.")
    else:
        c1, c2 = st.columns(2)
        with c1:
            if "Gender" in flt.columns and flt["Gender"].notna().any():
                st.plotly_chart(
                    px.pie(flt, names="Gender", title="Gender Distribution",
                            hole=0.4, color_discrete_sequence=px.colors.qualitative.Set2),
                    use_container_width=True)
        with c2:
            if "Age Group" in flt.columns and flt["Age Group"].notna().any():
                vc = flt["Age Group"].value_counts().sort_index().reset_index()
                vc.columns = ["Age Group", "Count"]
                st.plotly_chart(
                    px.bar(vc, x="Age Group", y="Count", title="Age Groups",
                            color="Age Group",
                            color_discrete_sequence=px.colors.qualitative.Vivid),
                    use_container_width=True)

        c3, c4 = st.columns(2)
        with c3:
            if "Age" in flt.columns and flt["Age"].notna().any():
                st.plotly_chart(
                    px.histogram(flt, x="Age", nbins=25, title="Age Distribution",
                                  color_discrete_sequence=["#e63946"]),
                    use_container_width=True)
        with c4:
            if "Is Admitted" in flt.columns and flt["Is Admitted"].notna().any():
                vc = flt["Is Admitted"].value_counts().reset_index()
                vc.columns = ["Status", "Count"]
                st.plotly_chart(
                    px.pie(vc, names="Status", values="Count",
                            title="Admission Status", hole=0.5,
                            color_discrete_sequence=px.colors.qualitative.Pastel),
                    use_container_width=True)

        if "Created At" in flt.columns and flt["Created At"].notna().any():
            t = flt.copy()
            t["_dt"] = pd.to_datetime(t["Created At"], errors="coerce")
            t["Hour"] = t["_dt"].dt.hour
            c5, c6 = st.columns(2)
            with c5:
                hr = t["Hour"].value_counts().sort_index().reset_index()
                hr.columns = ["Hour", "Count"]
                st.plotly_chart(
                    px.bar(hr, x="Hour", y="Count",
                            title="Patient Arrivals by Hour",
                            color_discrete_sequence=["#2a9d8f"]),
                    use_container_width=True)
            with c6:
                if "Created By Id" in t.columns:
                    sb = t["Created By Id"].value_counts().head(10).reset_index()
                    sb.columns = ["Staff ID", "Registrations"]
                    st.plotly_chart(
                        px.bar(sb, x="Staff ID", y="Registrations",
                                title="Registrations per Staff",
                                color="Registrations",
                                color_continuous_scale="Teal"),
                        use_container_width=True)


# ------------------------------------------------------------
# TAB 2 — CONSULTATIONS
# ------------------------------------------------------------
with tab2:
    if cons.empty:
        st.info("No consultation data in current filter.")
    else:
        if "Diagnosis" in cons.columns and cons["Diagnosis"].notna().any():
            top = cons["Diagnosis"].value_counts().head(15).reset_index()
            top.columns = ["Diagnosis", "Count"]
            st.plotly_chart(
                px.bar(top, x="Count", y="Diagnosis", orientation="h",
                        title="Top 15 Diagnoses",
                        color="Count", color_continuous_scale="Magma"),
                use_container_width=True)

            if not flt.empty and "Gender" in flt.columns:
                merged = cons.merge(flt[["Id", "Gender", "Age Group"]],
                                     left_on="Patient Id", right_on="Id",
                                     how="left", suffixes=("", "_p"))
                top10 = top.head(10)["Diagnosis"].tolist()
                sub = merged[merged["Diagnosis"].isin(top10)]

                if "Gender" in merged.columns:
                    ct = sub.groupby(["Diagnosis", "Gender"]).size().reset_index(name="Count")
                    if not ct.empty:
                        st.plotly_chart(
                            px.bar(ct, x="Count", y="Diagnosis", color="Gender",
                                    orientation="h", barmode="stack",
                                    title="Top 10 Diagnoses by Gender",
                                    color_discrete_sequence=px.colors.qualitative.Set2),
                            use_container_width=True)

                if "Age Group" in merged.columns:
                    ct2 = sub.groupby(["Diagnosis", "Age Group"]).size().reset_index(name="Count")
                    if not ct2.empty:
                        st.plotly_chart(
                            px.bar(ct2, x="Count", y="Diagnosis", color="Age Group",
                                    orientation="h", barmode="stack",
                                    title="Top 10 Diagnoses by Age Group",
                                    color_discrete_sequence=px.colors.qualitative.Vivid),
                            use_container_width=True)

        refs = [c for c in ["Refer To Pharmacy", "Refer To Laboratory",
                             "Refer To Optician", "Refer To Specialist"]
                if c in cons.columns]
        if refs:
            def pos(x):
                return int(x.sum()) if x.dtype == bool else \
                       int(x.astype(str).str.lower().isin(["true", "yes", "1"]).sum())
            counts = cons[refs].apply(pos).reset_index()
            counts.columns = ["Referral", "Count"]
            st.plotly_chart(
                px.bar(counts, x="Referral", y="Count",
                        title="Referral Distribution",
                        color="Referral",
                        color_discrete_sequence=px.colors.qualitative.Prism),
                use_container_width=True)

        if "Created At" in cons.columns and cons["Created At"].notna().any():
            c2 = cons.copy()
            c2["_dt"] = pd.to_datetime(c2["Created At"], errors="coerce")
            c2["Hour"] = c2["_dt"].dt.hour
            hr = c2["Hour"].value_counts().sort_index().reset_index()
            hr.columns = ["Hour", "Count"]
            st.plotly_chart(
                px.bar(hr, x="Hour", y="Count", title="Consultations by Hour",
                        color_discrete_sequence=["#f4a261"]),
                use_container_width=True)

        if "Created By Id" in cons.columns:
            sw = cons["Created By Id"].value_counts().head(15).reset_index()
            sw.columns = ["Staff ID", "Consultations"]
            st.plotly_chart(
                px.bar(sw, x="Staff ID", y="Consultations",
                        title="Staff Workload",
                        color="Consultations", color_continuous_scale="Blues"),
                use_container_width=True)


# ------------------------------------------------------------
# TAB 3 — LAB TESTS
# ------------------------------------------------------------
with tab3:
    if lab.empty:
        st.info("No lab test data in current filter.")
    else:
        lab_cols = [c for c in ["Malaria Parasite", "Random Blood Sugar", "Hbsag"]
                    if c in lab.columns]
        if lab_cols:
            cols = st.columns(min(3, len(lab_cols)))
            for i, col in enumerate(lab_cols):
                with cols[i % len(cols)]:
                    vc = lab[col].value_counts(dropna=False).reset_index()
                    vc.columns = ["Result", "Count"]
                    st.plotly_chart(
                        px.pie(vc, names="Result", values="Count",
                                title=col, hole=0.4,
                                color_discrete_sequence=px.colors.qualitative.Set3),
                        use_container_width=True)

        if "Completed" in lab.columns:
            vc = lab["Completed"].value_counts().reset_index()
            vc.columns = ["Status", "Count"]
            st.plotly_chart(
                px.bar(vc, x="Status", y="Count", title="Test Completion Status",
                        color="Status",
                        color_discrete_sequence=px.colors.qualitative.Set1),
                use_container_width=True)

        if {"Created At", "Completed At"}.issubset(lab.columns):
            tat = (pd.to_datetime(lab["Completed At"], errors="coerce") -
                   pd.to_datetime(lab["Created At"], errors="coerce")).dt.total_seconds() / 3600
            tat = tat.dropna()
            if not tat.empty:
                st.plotly_chart(
                    px.histogram(x=tat, nbins=25,
                                  title="Lab Test Turnaround Time (hours)",
                                  color_discrete_sequence=["#457b9d"]),
                    use_container_width=True)

        if "Malaria Parasite" in lab.columns and not flt.empty and "Age Group" in flt.columns:
            m = lab.merge(flt[["Id", "Age Group"]], left_on="Patient Id",
                           right_on="Id", how="left")
            m["_pos"] = m["Malaria Parasite"].astype(str).str.lower().isin(
                ["positive", "pos", "reactive", "+", "1", "true"])
            grp = m.groupby("Age Group")["_pos"].mean().mul(100).round(1).reset_index()
            grp.columns = ["Age Group", "Positive Rate (%)"]
            if not grp.empty:
                st.plotly_chart(
                    px.bar(grp, x="Age Group", y="Positive Rate (%)",
                            title="Malaria Positive Rate by Age Group",
                            color="Positive Rate (%)", color_continuous_scale="Reds"),
                    use_container_width=True)


# ------------------------------------------------------------
# TAB 4 — OPTICAL
# ------------------------------------------------------------
with tab4:
    if opt.empty:
        st.info("No optical data in current filter.")
    else:
        c1, c2 = st.columns(2)
        with c1:
            if "Is Walk In" in opt.columns:
                vc = opt["Is Walk In"].value_counts().reset_index()
                vc.columns = ["Type", "Count"]
                st.plotly_chart(
                    px.pie(vc, names="Type", values="Count",
                            title="Walk-in vs Referred", hole=0.4,
                            color_discrete_sequence=["#88c999", "#f4a582"]),
                    use_container_width=True)
        with c2:
            if "Glasses Allocated" in opt.columns:
                vc = opt["Glasses Allocated"].value_counts().reset_index()
                vc.columns = ["Allocated", "Count"]
                st.plotly_chart(
                    px.bar(vc, x="Allocated", y="Count",
                            title="Glasses Allocated",
                            color="Allocated",
                            color_discrete_sequence=px.colors.qualitative.Set2),
                    use_container_width=True)

        c3, c4 = st.columns(2)
        with c3:
            if "Visual Acuity Left" in opt.columns:
                vc = opt["Visual Acuity Left"].value_counts().reset_index()
                vc.columns = ["Acuity", "Count"]
                st.plotly_chart(
                    px.bar(vc, x="Acuity", y="Count",
                            title="Visual Acuity – Left Eye",
                            color_discrete_sequence=["#457b9d"]),
                    use_container_width=True)
        with c4:
            if "Visual Acuity Right" in opt.columns:
                vc = opt["Visual Acuity Right"].value_counts().reset_index()
                vc.columns = ["Acuity", "Count"]
                st.plotly_chart(
                    px.bar(vc, x="Acuity", y="Count",
                            title="Visual Acuity – Right Eye",
                            color_discrete_sequence=["#2a9d8f"]),
                    use_container_width=True)

        if "Glasses Type" in opt.columns:
            vc = opt["Glasses Type"].value_counts().reset_index()
            vc.columns = ["Type", "Count"]
            st.plotly_chart(
                px.bar(vc, x="Type", y="Count", title="Glasses Types",
                        color="Type",
                        color_discrete_sequence=px.colors.qualitative.Pastel),
                use_container_width=True)

        if "Refractive Error" in opt.columns and opt["Refractive Error"].notna().any():
            vc = opt["Refractive Error"].value_counts().reset_index()
            vc.columns = ["Error", "Count"]
            st.plotly_chart(
                px.bar(vc, x="Error", y="Count", title="Refractive Errors",
                        color="Error",
                        color_discrete_sequence=px.colors.qualitative.Vivid),
                use_container_width=True)


# ------------------------------------------------------------
# TAB 5 — PHARMACY
# ------------------------------------------------------------
with tab5:
    if ph.empty and drugs.empty:
        st.info("No pharmacy data in current filter.")
    else:
        if not ph.empty and "Drug Name" in ph.columns:
            names = filter_real_drugs(ph["Drug Name"])
            top = names.value_counts().head(15).reset_index()
            top.columns = ["Drug", "Count"]
            if not top.empty:
                st.plotly_chart(
                    px.bar(top, x="Count", y="Drug", orientation="h",
                            title="Top 15 Dispensed Drugs",
                            color="Count", color_continuous_scale="YlOrRd"),
                    use_container_width=True)

        c1, c2 = st.columns(2)
        with c1:
            if not ph.empty and "Dispensed" in ph.columns:
                vc = ph["Dispensed"].value_counts().reset_index()
                vc.columns = ["Status", "Count"]
                st.plotly_chart(
                    px.pie(vc, names="Status", values="Count", hole=0.4,
                            title="Pharmacy Dispense Rate",
                            color_discrete_sequence=px.colors.qualitative.Set2),
                    use_container_width=True)
        with c2:
            if not ph.empty and "Drug Name" in ph.columns:
                names = filter_real_drugs(ph["Drug Name"])
                top10 = names.value_counts().head(10)
                if not top10.empty:
                    st.plotly_chart(
                        px.bar(x=top10.values, y=top10.index, orientation="h",
                                title="Top 10 Drugs by Volume",
                                color=top10.values, color_continuous_scale="YlOrRd"),
                        use_container_width=True)

        if not drugs.empty and "Category" in drugs.columns:
            cat = drugs.groupby("Category").size().reset_index(name="Count")
            st.plotly_chart(
                px.bar(cat, x="Category", y="Count", title="Drugs by Category",
                        color="Category",
                        color_discrete_sequence=px.colors.qualitative.Bold),
                use_container_width=True)

        if not drugs.empty and {"Quantity", "Reorder Level", "Category"}.issubset(drugs.columns):
            d = drugs.copy()
            d["Stock Status"] = np.where(d["Quantity"] <= d["Reorder Level"],
                                          "Reorder", "OK")
            st.plotly_chart(
                px.bar(d, x="Category", y="Quantity", color="Stock Status",
                        barmode="group",
                        title="Stock by Category (with Status)",
                        color_discrete_map={"OK": "#2a9d8f", "Reorder": "#e63946"}),
                use_container_width=True)

            low = d[d["Stock Status"] == "Reorder"]
            if not low.empty:
                st.warning(f"⚠️ {len(low)} drug(s) need reordering")
                cols_to_show = [c for c in ["Name", "Category", "Quantity", "Reorder Level"]
                                if c in low.columns]
                st.dataframe(low[cols_to_show], use_container_width=True)


# ------------------------------------------------------------
# TAB 6 — VITALS
# ------------------------------------------------------------
with tab6:
    if nurse.empty:
        st.info("No nursing/vitals data in current filter.")
    else:
        vitals = [c for c in ["Blood Pressure Systolic", "Blood Pressure Diastolic",
                               "Pulse Rate", "Temperature",
                               "Respiratory Rate", "Oxygen Saturation"]
                  if c in nurse.columns and nurse[c].notna().any()]

        for i in range(0, len(vitals), 2):
            cols = st.columns(2)
            for j, col in enumerate(vitals[i:i+2]):
                with cols[j]:
                    st.plotly_chart(
                        px.histogram(nurse, x=col, nbins=25,
                                      title=f"Distribution – {col}",
                                      color_discrete_sequence=["#457b9d"]),
                        use_container_width=True)

        if vitals:
            melted = nurse[vitals].melt(var_name="Vital", value_name="Value").dropna()
            st.plotly_chart(
                px.box(melted, x="Vital", y="Value", color="Vital",
                        title="Vitals Comparison (Box Plots)",
                        color_discrete_sequence=px.colors.qualitative.Set3),
                use_container_width=True)

        if len(vitals) >= 2:
            corr = nurse[vitals].corr().round(2)
            st.plotly_chart(
                px.imshow(corr, text_auto=True, aspect="auto",
                           color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
                           title="Vitals Correlation Matrix"),
                use_container_width=True)

        c1, c2 = st.columns(2)
        with c1:
            if "Biohazard Risk" in nurse.columns:
                vc = nurse["Biohazard Risk"].value_counts().reset_index()
                vc.columns = ["Risk", "Count"]
                st.plotly_chart(
                    px.pie(vc, names="Risk", values="Count", hole=0.4,
                            title="Biohazard Risk",
                            color_discrete_sequence=px.colors.qualitative.Set1),
                    use_container_width=True)
        with c2:
            if "Isolation Required" in nurse.columns:
                vc = nurse["Isolation Required"].value_counts().reset_index()
                vc.columns = ["Isolation", "Count"]
                st.plotly_chart(
                    px.pie(vc, names="Isolation", values="Count", hole=0.4,
                            title="Isolation Required",
                            color_discrete_sequence=px.colors.qualitative.Set2),
                    use_container_width=True)


# ------------------------------------------------------------
# TAB 7 — TRENDS
# ------------------------------------------------------------
with tab7:
    st.markdown("### 📈 Outreach Activity Breakdown")

    services = {
        "Registrations": len(flt),
        "Consultations": len(cons),
        "Lab Tests": len(lab),
        "Optical": len(opt),
        "Pharmacy": len(ph),
        "Nursing": len(nurse),
    }
    sdf = pd.DataFrame(list(services.items()), columns=["Service", "Count"])
    st.plotly_chart(
        px.bar(sdf, x="Service", y="Count", text="Count",
                title="Activity Volume by Service",
                color="Service",
                color_discrete_sequence=px.colors.qualitative.Bold),
        use_container_width=True)

    if not cons.empty:
        cpp = cons.groupby("Patient Id").size().reset_index(name="Visits")
        st.plotly_chart(
            px.histogram(cpp, x="Visits", nbins=15,
                          title="Consultations per Patient",
                          color_discrete_sequence=["#6a4c93"]),
            use_container_width=True)

    if not flt.empty and "Age Group" in flt.columns and "Gender" in flt.columns:
        pyr = flt.groupby(["Age Group", "Gender"]).size().reset_index(name="Count")
        st.plotly_chart(
            px.bar(pyr, x="Count", y="Age Group", color="Gender",
                    orientation="h", barmode="relative",
                    title="Age Pyramid by Gender",
                    color_discrete_sequence=px.colors.qualitative.Set2),
            use_container_width=True)


# ------------------------------------------------------------
# TAB 8 — TABLES (PII stripped)
# ------------------------------------------------------------
with tab8:
    st.markdown("### 📋 Data Tables")
    st.caption("⚠️ Patient bio-data (names, phone, email, address, DOB) is hidden for privacy. "
               "Only clinical and operational records are shown.")

    def safe_view(df, max_rows=500):
        return hide_pii(df).head(max_rows) if df is not None and not df.empty else df

    t8a, t8b, t8c = st.tabs(["🩺 Consultations", "🧪 Lab Tests", "💊 Pharmacy"])

    with t8a:
        st.caption(f"{len(cons)} rows (showing up to 500)")
        if cons.empty:
            st.info("No consultation records.")
        else:
            st.dataframe(safe_view(cons), use_container_width=True, height=400)

    with t8b:
        st.caption(f"{len(lab)} rows (showing up to 500)")
        if lab.empty:
            st.info("No lab test records.")
        else:
            st.dataframe(safe_view(lab), use_container_width=True, height=400)

    with t8c:
        st.caption(f"{len(ph)} rows (showing up to 500)")
        if ph.empty:
            st.info("No pharmacy records.")
        else:
            st.dataframe(safe_view(ph), use_container_width=True, height=400)


# ------------------------------------------------------------
# TAB 9 — PREDICT (Clinical Decision Support)
# ------------------------------------------------------------
with tab9:
    st.markdown("### 🔮 Clinical Decision Support")
    st.caption(
        "Enter patient vitals for instant risk assessment. "
        "Uses trained ML models when available; falls back to clinical heuristics otherwise."
    )

    with st.expander("🔌 API Status", expanded=False):
        try:
            r = httpx.get(f"{API_URL}/health", timeout=3.0)
            if r.status_code == 200:
                info = r.json()
                st.success(f"✅ Connected to {API_URL}")
                st.json(info)
            else:
                st.error(f"⚠️ API returned {r.status_code}")
        except Exception as e:
            st.error(f"❌ Cannot reach API at {API_URL}")
            st.code(f"""
# Start the API locally:
uvicorn api.app:app --reload --port 8000

# Or deploy to Render and set the URL in Streamlit secrets:
API_URL = "https://my-api.onrender.com"

# Current error: {e}
""", language="bash")

    st.markdown("---")

    col_malaria, col_admission = st.columns(2)

    with col_malaria:
        st.markdown("#### 🦟 Malaria Risk Assessment")

        with st.form("malaria_form"):
            m_age = st.number_input("Age", min_value=0, max_value=120,
                                     value=25, step=1, key="m_age")
            m_gender = st.selectbox("Gender", ["Female", "Male", "Other"],
                                     key="m_gender")
            m_temp = st.slider("Temperature (°C)", 35.0, 42.0, 37.5,
                                step=0.1, key="m_temp")
            m_pulse = st.slider("Pulse (bpm)", 40, 200, 80,
                                 step=1, key="m_pulse")
            m_sys = st.slider("Systolic BP (mmHg)", 70, 220, 120,
                               step=1, key="m_sys")
            m_dia = st.slider("Diastolic BP (mmHg)", 40, 140, 80,
                               step=1, key="m_dia")

            submit_m = st.form_submit_button("🦟 Assess Malaria Risk",
                                              use_container_width=True,
                                              type="primary")

        if submit_m:
            payload = {
                "Age": m_age, "Gender": m_gender,
                "avg_temp": m_temp, "avg_pulse": m_pulse,
                "avg_sys": m_sys, "avg_dia": m_dia,
            }
            with st.spinner("Assessing..."):
                result = call_api("/predict/malaria", payload)

            if result and "error" not in result:
                prob = result["probability"]
                pred = result["prediction"]

                st.metric(
                    "Malaria Risk",
                    f"{prob:.1%}",
                    delta="Positive (likely)" if pred else "Negative (likely)",
                    delta_color="inverse" if pred else "normal",
                )

                if pred:
                    st.error(f"🔴 {result['label']}")
                else:
                    st.success(f"🟢 {result['label']}")

                st.progress(min(prob, 1.0))

                with st.expander("📋 Prediction details", expanded=False):
                    st.json(result)

                st.markdown("##### 🩺 Interpretation")
                if prob >= 0.7:
                    st.markdown("**High risk** — recommend immediate malaria RDT or microscopy.")
                elif prob >= 0.4:
                    st.markdown("**Moderate risk** — consider testing; monitor symptoms.")
                else:
                    st.markdown("**Low risk** — routine observation; test if symptoms develop.")
            elif result and "error" in result:
                st.error(f"❌ {result['error']}")

    with col_admission:
        st.markdown("#### 🏥 Admission Risk Assessment")

        with st.form("admission_form"):
            a_age = st.number_input("Age", min_value=0, max_value=120,
                                     value=45, step=1, key="a_age")
            a_gender = st.selectbox("Gender", ["Female", "Male", "Other"],
                                     key="a_gender")
            a_cons = st.number_input("Number of consultations today",
                                      min_value=0, max_value=20, value=1,
                                      step=1, key="a_cons")
            a_sys = st.slider("Systolic BP (mmHg)", 70, 220, 125,
                               step=1, key="a_sys")
            a_dia = st.slider("Diastolic BP (mmHg)", 40, 140, 82,
                               step=1, key="a_dia")
            a_temp = st.slider("Temperature (°C)", 35.0, 42.0, 37.0,
                                step=0.1, key="a_temp")
            a_pulse = st.slider("Pulse (bpm)", 40, 200, 78,
                                 step=1, key="a_pulse")
            a_spo2 = st.slider("SpO₂ (%)", 70, 100, 97,
                                step=1, key="a_spo2")

            submit_a = st.form_submit_button("🏥 Assess Admission Risk",
                                              use_container_width=True,
                                              type="primary")

        if submit_a:
            payload = {
                "Age": a_age, "Gender": a_gender,
                "n_consultations": a_cons,
                "avg_sys": a_sys, "avg_dia": a_dia,
                "avg_temp": a_temp, "avg_pulse": a_pulse,
                "avg_spo2": a_spo2,
            }
            with st.spinner("Assessing..."):
                result = call_api("/predict/admission", payload)

            if result and "error" not in result:
                prob = result["probability"]
                pred = result["prediction"]

                st.metric(
                    "Admission Risk",
                    f"{prob:.1%}",
                    delta="Likely Admitted" if pred else "Likely Outpatient",
                    delta_color="inverse" if pred else "normal",
                )

                if pred:
                    st.error(f"🔴 {result['label']}")
                else:
                    st.success(f"🟢 {result['label']}")

                st.progress(min(prob, 1.0))

                with st.expander("📋 Prediction details", expanded=False):
                    st.json(result)

                st.markdown("##### 🩺 Interpretation")
                if prob >= 0.7:
                    st.markdown("**High risk** — recommend admission or close observation.")
                elif prob >= 0.4:
                    st.markdown("**Moderate risk** — monitor vitals; reassess in 2–4 hours.")
                else:
                    st.markdown("**Low risk** — suitable for outpatient management.")
            elif result and "error" in result:
                st.error(f"❌ {result['error']}")

    st.markdown("---")
    with st.expander("📊 Batch prediction (paste multiple patients)", expanded=False):
        st.caption(
            "Enter one patient per line. Format: "
            "`age,gender,temp,pulse,systolic,diastolic`"
        )
        batch_text = st.text_area(
            "Patients",
            height=150,
            placeholder="8,Female,39.2,115,100,65\n45,Male,37.0,78,125,82\n...",
        )
        if st.button("Run batch malaria prediction"):
            lines = [l.strip() for l in batch_text.strip().split("\n") if l.strip()]
            if not lines:
                st.warning("No data entered.")
            else:
                rows = []
                progress = st.progress(0)
                for i, line in enumerate(lines):
                    try:
                        parts = [p.strip() for p in line.split(",")]
                        payload = {
                            "Age": float(parts[0]),
                            "Gender": parts[1],
                            "avg_temp": float(parts[2]),
                            "avg_pulse": float(parts[3]),
                            "avg_sys": float(parts[4]),
                            "avg_dia": float(parts[5]),
                        }
                        r = call_api("/predict/malaria", payload)
                        if r and "error" not in r:
                            rows.append({
                                "Input": line,
                                "Risk": f"{r['probability']:.1%}",
                                "Prediction": r["label"],
                            })
                        else:
                            rows.append({"Input": line, "Risk": "—",
                                          "Prediction": r.get("error", "Failed")})
                    except Exception as e:
                        rows.append({"Input": line, "Risk": "—",
                                      "Prediction": f"Parse error: {e}"})
                    progress.progress((i + 1) / len(lines))

                if rows:
                    st.dataframe(rows, use_container_width=True)
                    df_batch = pd.DataFrame(rows)
                    st.download_button(
                        "⬇️ Download results as CSV",
                        df_batch.to_csv(index=False),
                        file_name="batch_predictions.csv",
                        mime="text/csv",
                    )


# ------------------------------------------------------------
# TAB 10 — ASK BGIE AI (Dual-Mode Chat + Persistent Charts)
#   • Outreach Data mode — answers ONLY from today's data
#   • General Knowledge mode — answers anything; charts via ```chart``` JSON
#   • Provider: Local Ollama first, Ollama Cloud fallback
#   • Layout: responses stream ABOVE a pinned input at the bottom
#   • Charts persist in chat history
#   • Auto-scrolls to newest message
# ------------------------------------------------------------
with tab10:
    st.markdown("### 🤖 Ask BGIE AI")
    st.caption(
        "Ask anything — about today's outreach data OR general questions. "
        "Try: *\"Plot the top 5 diagnoses\"* or *\"Show gender split as pie\"*."
    )

    # Sticky input styling
    st.markdown("""
    <style>
        div[data-testid="stChatInput"] {
            position: sticky;
            bottom: 0;
            background: var(--background-color, #fff);
            padding-top: 8px;
            z-index: 10;
            border-top: 1px solid rgba(0,0,0,0.08);
        }
    </style>
    """, unsafe_allow_html=True)

    # ============================================================
    # Local imports
    # ============================================================
    import json as _json
    import re as _re
    import plotly.express as _px
    import pandas as _pd

    # ============================================================
    # Provider detection
    # ============================================================
    @st.cache_data(ttl=60, show_spinner=False)
    def _local_ollama_models():
        try:
            result = ollama.list()
            return [m.get("name", m.get("model", "unknown"))
                    for m in result.get("models", [])]
        except Exception:
            return []

    def _cloud_config():
        def _get(key, default=""):
            try:
                return st.secrets.get(key, default)
            except Exception:
                import os
                return os.environ.get(key, default)

        return {
            "enabled": str(_get("CLOUD_ENABLED", "false")).lower() == "true",
            "url":     _get("CLOUD_URL", "https://ollama.com/v1"),
            "api_key": _get("CLOUD_API_KEY", ""),
            "model":   _get("CLOUD_MODEL", "gpt-oss:20b"),
        }

    local_models = _local_ollama_models()
    cloud_cfg = _cloud_config()
    has_local = len(local_models) > 0
    has_cloud = bool(
        cloud_cfg["enabled"] and cloud_cfg["url"] and cloud_cfg["api_key"]
    )

    # ============================================================
    # No provider — setup instructions
    # ============================================================
    if not has_local and not has_cloud:
        st.warning(
            "⚠️ **No LLM provider available.** "
            "Set up local Ollama or Ollama Cloud."
        )
        st.code("""
# Local
ollama serve
ollama pull llama3.2

# Or cloud (Streamlit Cloud → Settings → Secrets)
CLOUD_ENABLED = "true"
CLOUD_URL     = "https://ollama.com/v1"
CLOUD_API_KEY = "my-ollama-cloud-key"
CLOUD_MODEL   = "gpt-oss:20b"
""", language="bash")
    else:
        # ============================================================
        # Controls — Provider, Model, Mode
        # ============================================================
        providers = []
        if has_local:
            providers.append("🖥️ Local Ollama (fast, private)")
        if has_cloud:
            providers.append("☁️ Ollama Cloud (works everywhere)")

        col_provider, col_model = st.columns([1, 1])

        with col_provider:
            provider = st.radio(
                "⚡ Provider",
                options=providers,
                index=0,
                horizontal=True,
            )
            use_local = provider.startswith("🖥️")

        with col_model:
            if use_local:
                selected_model = st.selectbox(
                    "🧠 Model",
                    options=local_models,
                    index=next(
                        (i for i, m in enumerate(local_models)
                         if "llama3.2" in m.lower()),
                        0,
                    ),
                )
            else:
                cloud_models = [
                    "gpt-oss:20b", "gpt-oss:120b",
                    "glm-5.3-flash", "deepseek-v4.1-flash",
                    "nemotron-3-nano:30b", "gemma4:31b",
                    "kimi-k2.6", "mistral-large-3:675b",
                    "glm-5.3", "qwen3.5:397b",
                ]
                default_idx = 0
                if cloud_cfg["model"] in cloud_models:
                    default_idx = cloud_models.index(cloud_cfg["model"])
                selected_model = st.selectbox(
                    "🧠 Model",
                    options=cloud_models,
                    index=default_idx,
                )

        mode = st.radio(
            "🎛️ Mode",
            options=["📊 Outreach Data", "🌍 General Knowledge"],
            index=0,
            horizontal=True,
            help=(
                "**Outreach Data** — only from today's numbers. "
                "**General Knowledge** — anything + full chart support."
            ),
        )
        is_data_mode = mode.startswith("📊")

        st.caption(
            f"**Active:** {provider.split('(')[0].strip()} · "
            f"Model: `{selected_model}` · "
            f"Mode: **{mode}**"
        )

        # ============================================================
        # Context builder
        # ============================================================
        def _build_context():
            lines = ["Today's outreach summary:"]
            lines.append(f"- Total patients registered: {len(flt)}")
            lines.append(f"- Total consultations: {len(cons)}")
            lines.append(f"- Total lab tests: {len(lab)}")
            lines.append(f"- Total optical assessments: {len(opt)}")
            lines.append(f"- Total pharmacy orders: {len(ph)}")
            lines.append(f"- Total nursing assessments: {len(nurse)}")

            if not flt.empty and "Gender" in flt.columns:
                lines.append(
                    f"- Gender distribution: "
                    f"{flt['Gender'].value_counts().to_dict()}"
                )
            if (not flt.empty and "Age" in flt.columns
                    and flt["Age"].notna().any()):
                lines.append(f"- Average age: {flt['Age'].mean():.1f} years")
            if not flt.empty and "Age Group" in flt.columns:
                lines.append(
                    f"- Age groups: "
                    f"{flt['Age Group'].value_counts().sort_index().to_dict()}"
                )
            if not cons.empty and "Diagnosis" in cons.columns:
                top_dx = cons["Diagnosis"].value_counts().head(10)
                lines.append(f"- Top 10 diagnoses: {top_dx.to_dict()}")
            if not lab.empty and "Malaria Parasite" in lab.columns:
                pos = lab["Malaria Parasite"].astype(str).str.lower().isin(
                    ["positive", "pos", "reactive", "+", "1", "true"]
                ).sum()
                lines.append(
                    f"- Malaria positive tests: {pos} out of {len(lab)}"
                )
            if not ph.empty and "Drug Name" in ph.columns:
                try:
                    from main_analysis import filter_real_drugs
                    names = filter_real_drugs(ph["Drug Name"])
                    top_drugs = names.value_counts().head(10)
                    lines.append(
                        f"- Top 10 dispensed drugs: {top_drugs.to_dict()}"
                    )
                except Exception:
                    pass
            if not nurse.empty:
                for v in ["Blood Pressure Systolic", "Temperature",
                          "Pulse Rate", "Oxygen Saturation"]:
                    if v in nurse.columns and nurse[v].notna().any():
                        lines.append(f"- Mean {v}: {nurse[v].mean():.1f}")
            return "\n".join(lines)

        # ============================================================
        # Session state
        # ============================================================
        if "chat_messages" not in st.session_state:
            st.session_state.chat_messages = []

        if "chat_mode" not in st.session_state:
            st.session_state.chat_mode = mode
        if st.session_state.chat_mode != mode:
            st.session_state.chat_messages = []
            st.session_state.chat_mode = mode

        with st.expander("📋 View data context sent to the AI", expanded=False):
            st.code(_build_context(), language="text")

        # ============================================================
        # Chart parsing + rendering
        # ============================================================
        _CHART_BLOCK_RE = _re.compile(
            r"```(?:chart|json)\s*(\{.*?\})\s*```",
            _re.DOTALL | _re.IGNORECASE,
        )

        def _try_repair_json(raw: str):
            s = raw.strip()
            s = _re.sub(r",\s*([}\]])", r"\1", s)
            open_braces = s.count("{") - s.count("}")
            open_brackets = s.count("[") - s.count("]")
            if s.count('"') % 2 == 1:
                s += '"'
            s += "]" * max(0, open_brackets)
            s += "}" * max(0, open_braces)
            try:
                data = _json.loads(s)
                if isinstance(data, dict) and "type" in data:
                    labels = data.get("labels") or []
                    values = data.get("values") or []
                    if labels and values:
                        n = min(len(labels), len(values))
                        data["labels"] = labels[:n]
                        data["values"] = values[:n]
                        return data
            except Exception:
                pass
            return None

        def _extract_chart_block(text: str):
            for m in _CHART_BLOCK_RE.finditer(text):
                raw = m.group(1)
                try:
                    data = _json.loads(raw)
                    if isinstance(data, dict) and "type" in data:
                        return data
                except Exception:
                    repaired = _try_repair_json(raw)
                    if repaired:
                        return repaired

            pattern = _re.compile(
                r"```(?:chart|json)\s*(\{.*?)(?=```|\Z)",
                _re.DOTALL | _re.IGNORECASE,
            )
            for m in pattern.finditer(text):
                repaired = _try_repair_json(m.group(1))
                if repaired:
                    return repaired

            pattern2 = _re.compile(
                r'\{[^{}]*"type"\s*:\s*"[^"]+"[^{}]*"labels"[^{}]*\}',
                _re.DOTALL,
            )
            for m in pattern2.finditer(text):
                try:
                    data = _json.loads(m.group(0))
                    if isinstance(data, dict) and "type" in data:
                        return data
                except Exception:
                    continue

            return None

        def _strip_chart_block(text: str) -> str:
            s = _CHART_BLOCK_RE.sub("", text)
            s = _re.sub(
                r"```(?:chart|json)\s*\{.*?(?=```|\Z).*?(?:```|\Z)",
                "", s, flags=_re.DOTALL | _re.IGNORECASE,
            )
            return s.strip()

        def _render_chart(spec: dict):
            ctype = str(spec.get("type", "")).lower()
            title = spec.get("title", "Chart")
            labels = spec.get("labels") or []
            values = spec.get("values") or []

            if not labels or not values or len(labels) != len(values):
                st.warning(
                    f"⚠️ Chart data mismatch "
                    f"(labels={len(labels)}, values={len(values)})."
                )
                return

            df_chart = _pd.DataFrame({"Category": labels, "Value": values})

            try:
                if ctype == "bar":
                    fig = _px.bar(df_chart, x="Category", y="Value",
                                  title=title, color="Value",
                                  color_continuous_scale="Viridis",
                                  text="Value")
                elif ctype == "barh":
                    fig = _px.bar(df_chart, x="Value", y="Category",
                                  orientation="h", title=title,
                                  color="Value",
                                  color_continuous_scale="Viridis",
                                  text="Value")
                elif ctype == "pie":
                    fig = _px.pie(df_chart, names="Category", values="Value",
                                  title=title, hole=0.0)
                elif ctype == "donut":
                    fig = _px.pie(df_chart, names="Category", values="Value",
                                  title=title, hole=0.5)
                elif ctype == "line":
                    fig = _px.line(df_chart, x="Category", y="Value",
                                   title=title, markers=True)
                elif ctype == "scatter":
                    fig = _px.scatter(df_chart, x="Category", y="Value",
                                      title=title, color="Value",
                                      color_continuous_scale="Plasma")
                elif ctype == "histogram":
                    fig = _px.histogram(df_chart, x="Category", y="Value",
                                        title=title)
                else:
                    fig = _px.bar(df_chart, x="Category", y="Value",
                                  title=title, color="Value",
                                  color_continuous_scale="Viridis",
                                  text="Value")

                fig.update_layout(
                    title_font_size=16,
                    title_x=0.5,
                    margin=dict(l=20, r=20, t=60, b=20),
                )
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.error(f"Could not render chart: {e}")

        def _scroll_to_latest():
            """Injects JS to auto-scroll the page to the newest chat message."""
            st.markdown(
                """
                <script>
                    setTimeout(function() {
                        const containers = window.parent.document.querySelectorAll(
                            'div[data-testid="stChatMessage"]'
                        );
                        if (containers.length > 0) {
                            containers[containers.length - 1].scrollIntoView({
                                behavior: "smooth",
                                block: "center"
                            });
                        }
                    }, 150);
                </script>
                """,
                unsafe_allow_html=True,
            )

        # ============================================================
        # Streaming helpers
        # ============================================================
        def _stream_local(model, messages):
            stream = ollama.chat(
                model=model,
                messages=messages,
                stream=True,
                options={"temperature": 0.3, "num_predict": 1500},
            )
            for chunk in stream:
                piece = chunk.get("message", {}).get("content", "")
                if piece:
                    yield piece

        def _stream_cloud(model, messages, url, api_key):
            import httpx
            endpoint = url.rstrip("/") + "/chat/completions"
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            }
            payload = {
                "model": model,
                "messages": messages,
                "stream": True,
                "temperature": 0.3,
                "max_tokens": 1500,
            }
            with httpx.stream("POST", endpoint, headers=headers,
                              json=payload, timeout=180.0) as r:
                r.raise_for_status()
                for line in r.iter_lines():
                    if not line:
                        continue
                    if line.startswith("data: "):
                        line = line[6:]
                    if line.strip() == "[DONE]":
                        break
                    try:
                        data = _json.loads(line)
                        delta = (data.get("choices", [{}])[0]
                                     .get("delta", {})
                                     .get("content", ""))
                        if delta:
                            yield delta
                    except Exception:
                        continue

        # ============================================================
        # System prompts (with chart rules)
        # ============================================================
        _CHART_FORMAT = """```chart
{"type": "<TYPE>", "title": "<TITLE>", "labels": ["<L1>", "<L2>"], "values": [<V1>, <V2>]}
```"""

        _CHART_RULES = (
            "CHART OUTPUT — ABSOLUTE RULES:\n"
            "\n"
            "This application renders charts AUTOMATICALLY from a JSON spec that YOU produce.\n"
            "You are NOT a plain text assistant. You DO have chart rendering capability — "
            "via the JSON block format below. The dashboard executes your JSON.\n"
            "\n"
            "When the user asks to plot, chart, graph, show, or visualize anything, "
            "output EXACTLY ONE chart block in this JSON format:\n"
            "\n"
            "```chart\n"
            '{"type": "<TYPE>", "title": "<TITLE>", "labels": ["<L1>", "<L2>"], "values": [<V1>, <V2>]}\n'
            "```\n"
            "\n"
            "STRICTLY FORBIDDEN — you must NEVER:\n"
            "- Write Python code (no `import matplotlib`, no `plt.bar`, no `plt.show`)\n"
            "- Say \"I am text-based\" or \"I cannot display images/charts\"\n"
            "- Produce ASCII art or Markdown tables as a chart substitute\n"
            "- Tell the user to run the code themselves\n"
            "- Output MORE THAN ONE chart block per response\n"
            "\n"
            "The ONLY valid chart output is the ```chart``` JSON block above.\n"
            "\n"
            "Chart types:\n"
            "- \"bar\" — vertical bars (age groups, hours)\n"
            "- \"barh\" — horizontal bars (diagnoses, drugs)\n"
            "- \"pie\" — full pie (distributions)\n"
            "- \"donut\" — pie with center hole\n"
            "- \"line\" — time trends\n"
            "- \"scatter\" — correlations\n"
            "- \"histogram\" — value distributions\n"
            "\n"
            "Rules:\n"
            "1. labels and values MUST be equal length.\n"
            "2. Use REAL numbers from the CONTEXT for outreach data.\n"
            "3. For general requests (\"generate data\"), invent reasonable values.\n"
            "4. Add a short one-sentence intro before the block.\n"
            "5. Output ONE chart only."
        )

        def _data_mode_prompt():
            return (
                "You are a healthcare analytics assistant for the COREP "
                "annual medical outreach.\n\n"
                "STRICT MODE: Answer ONLY from the context below. "
                "Do not use outside knowledge.\n\n"
                "CONTEXT (today's outreach data):\n"
                + _build_context() + "\n\n"
                "RULES:\n"
                "1. Base all answers strictly on the context.\n"
                "2. If the answer isn't in the context, say: "
                "\"I don't have that information in today's data.\"\n"
                "3. Never invent patient names, IDs, or specific individuals.\n"
                "4. Include exact numbers when referencing data.\n"
                "5. Keep text concise (2-4 sentences) unless asked for detail.\n\n"
                + _CHART_RULES
            )

        def _general_mode_prompt():
            return (
                "You are a knowledgeable AI assistant for the COREP "
                "annual medical outreach.\n\n"
                "GENERAL MODE: Answer ANY question — programming, medicine, "
                "science, math, general knowledge — AND questions about "
                "today's outreach.\n\n"
                "TODAY'S OUTREACH CONTEXT (use when relevant):\n"
                + _build_context() + "\n\n"
                "RULES:\n"
                "1. Answer general questions naturally, accurately, helpfully.\n"
                "2. When referring to today's outreach, use numbers from context.\n"
                "3. Never invent patient names or IDs.\n"
                "4. Use markdown (bold, bullets, code blocks, tables) when helpful.\n"
                "5. For programming questions, include short ```python``` examples.\n"
                "6. For medical advice, add a brief disclaimer.\n"
                "7. Be concise.\n\n"
                + _CHART_RULES
            )

        # ============================================================
        # Layout — history → reply container → pinned input
        # ============================================================
        for msg in st.session_state.chat_messages:
            with st.chat_message(msg["role"]):
                stored_chart = _extract_chart_block(msg["content"])
                stored_text = _strip_chart_block(msg["content"])
                if stored_text:
                    st.markdown(stored_text)
                if stored_chart:
                    st.caption("📊 Chart")
                    _render_chart(stored_chart)

        reply_container = st.container()
        input_container = st.container()

        placeholder_text = (
            "Ask about today's outreach data…"
            if is_data_mode
            else "Ask anything — or try *Plot the top 5 diagnoses*"
        )
        with input_container:
            user_prompt = st.chat_input(placeholder_text)

        # ============================================================
        # Handle input
        # ============================================================
        if user_prompt:
            st.session_state.chat_messages.append(
                {"role": "user", "content": user_prompt}
            )
            with reply_container:
                with st.chat_message("user"):
                    st.markdown(user_prompt)

            system_prompt = (
                _data_mode_prompt() if is_data_mode else _general_mode_prompt()
            )

            full_messages = [
                {"role": "system", "content": system_prompt},
                *[
                    {"role": m["role"], "content": m["content"]}
                    for m in st.session_state.chat_messages
                ],
            ]

            with reply_container:
                with st.chat_message("assistant"):
                    placeholder = st.empty()
                    full_response = ""
                    try:
                        stream_iter = (
                            _stream_local(selected_model, full_messages)
                            if use_local
                            else _stream_cloud(
                                selected_model, full_messages,
                                cloud_cfg["url"], cloud_cfg["api_key"],
                            )
                        )

                        chunk_count = 0
                        for piece in stream_iter:
                            full_response += piece
                            visible = _strip_chart_block(full_response)
                            placeholder.markdown(visible + " ▌")
                            chunk_count += 1
                            if chunk_count % 10 == 0:
                                _scroll_to_latest()

                        chart_spec = _extract_chart_block(full_response)
                        text_part = _strip_chart_block(full_response)

                        placeholder.empty()
                        if text_part:
                            st.markdown(text_part)
                        if chart_spec:
                            st.caption("📊 AI-generated chart")
                            _render_chart(chart_spec)

                        st.session_state.chat_messages.append(
                            {"role": "assistant", "content": full_response}
                        )
                    except Exception as e:
                        full_response = f"❌ Error: {e}"
                        placeholder.error(full_response)
                        st.session_state.chat_messages.append(
                            {"role": "assistant", "content": full_response}
                        )

            _scroll_to_latest()
            time.sleep(0.2)
            st.rerun()

        # ============================================================
        # Footer actions
        # ============================================================
        col_a, col_b = st.columns([1, 4])
        with col_a:
            if st.session_state.chat_messages:
                if st.button("🗑️ Clear chat", use_container_width=True):
                    st.session_state.chat_messages = []
                    st.rerun()

        # ============================================================
        # Example prompts
        # ============================================================
        with st.expander("💡 Try asking…", expanded=False):
            if is_data_mode:
                st.markdown("""
**📊 Outreach Data mode:**
- *"How many patients were registered today?"*
- *"What is the gender split?"*
- *"Plot the top 5 diagnoses as a bar chart."*
- *"Show the gender split as a pie chart."*
- *"Chart consultations by hour."*
- *"Give me an executive summary."*
""")
            else:
                st.markdown("""
**🌍 General Knowledge mode:**
- *"What is Python programming?"*
- *"Explain machine learning in simple terms."*
- *"Plot the top 5 diagnoses from the outreach data."*
- *"Show gender split as a pie chart."*
- *"Chart the age groups."*
- *"Plot the top 10 drugs as horizontal bars."*
- *"Write a Python function to calculate BMI."*
- *"What are the symptoms of malaria?"*
""")


# ============================================================
# Main footer (branded)
# ============================================================
st.markdown("---")
st.markdown(
    f"""
    <div style="text-align:center; color:#666; font-size:0.85rem; line-height:1.6; padding:16px 0;">
        <strong style="color:#0b4a6f; font-size:1rem;">{BRAND_LINE_FULL}</strong><br>
        {POWERED_BY}<br>
        <em>{DEVELOPER_LINE_FULL}</em><br>
        <span style="font-size:0.75rem;">{COPYRIGHT}</span>
    </div>
    """,
    unsafe_allow_html=True,
)