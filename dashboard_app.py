"""
Streamlit dashboard for COREP data — EXPANDED, branded, PII-safe.
Optimized for single-day medical outreach data.
Run:  streamlit run dashboard_app.py
"""

import os
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from main_analysis import filter_real_drugs, _match_sheet, EXCEL_FILE
from branding import (
    BRAND_LINE, DEVELOPER_LINE, COPYRIGHT, POWERED_BY,
    APP_TITLE, TRADEMARK, LOGO_PATH, DEVELOPER_NAME, COMPANY_NAME,
)

st.set_page_config(
    page_title=APP_TITLE,
    layout="wide",
    page_icon="🏥",
    initial_sidebar_state="expanded",
)

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

    # Coerce IDs to Int64
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
# Outreach banner (branded)
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
        {DEVELOPER_LINE}
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

# Age
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

# Gender
ug = sorted(patient["Gender"].dropna().astype(str).unique()) \
     if not patient.empty and "Gender" in patient.columns else []
genders = st.sidebar.multiselect("Gender", options=ug, default=ug)

# Admission
if not patient.empty and "Is Admitted" in patient.columns:
    adm_vals = sorted(patient["Is Admitted"].dropna().astype(str).unique())
else:
    adm_vals = []
adm_sel = st.sidebar.multiselect("Admission status", options=adm_vals, default=adm_vals)

# Apply filters
flt = patient.copy()
if not flt.empty and "Age" in flt.columns and flt["Age"].notna().any():
    flt = flt[(flt["Age"] >= age_range[0]) & (flt["Age"] <= age_range[1])]
if genders and "Gender" in flt.columns:
    flt = flt[flt["Gender"].astype(str).isin(genders)]
if adm_sel and "Is Admitted" in flt.columns:
    flt = flt[flt["Is Admitted"].astype(str).isin(adm_sel)]

patient_ids = set(flt["Id"].dropna()) if "Id" in flt.columns else set()


# ============================================================
# Filter helper
# ============================================================
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

# ---- Sidebar branding ----
st.sidebar.markdown("---")
st.sidebar.markdown(
    f"""
    <div style="text-align:center; color:#666; font-size:0.8rem; line-height:1.5;">
        <strong style="color:#0b4a6f;">{BRAND_LINE}</strong><br>
        {DEVELOPER_LINE}<br>
        <span style="font-size:0.75rem;">{COPYRIGHT}</span>
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
])
tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = tabs


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


# ============================================================
# Main footer (branded)
# ============================================================
st.markdown("---")
st.markdown(
    f"""
    <div style="text-align:center; color:#666; font-size:0.85rem; line-height:1.6; padding:16px 0;">
        <strong style="color:#0b4a6f; font-size:1rem;">{BRAND_LINE}</strong><br>
        {POWERED_BY}<br>
        <em>{DEVELOPER_LINE}</em><br>
        <span style="font-size:0.75rem;">{COPYRIGHT}</span>
    </div>
    """,
    unsafe_allow_html=True,
)