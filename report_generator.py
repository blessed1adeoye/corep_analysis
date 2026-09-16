"""
Generates a styled HTML report (and optional PDF) from the analysis outputs.
Run AFTER main_analysis.py.
"""

import os
import base64
from datetime import datetime
import pandas as pd
from jinja2 import Template

import main_analysis as analysis

CHART_DIR = "charts"
REPORT_DIR = "reports"
os.makedirs(REPORT_DIR, exist_ok=True)


def img_to_base64(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def collect_charts():
    if not os.path.isdir(CHART_DIR):
        return []
    files = sorted(f for f in os.listdir(CHART_DIR) if f.lower().endswith(".png"))
    return [(f, os.path.join(CHART_DIR, f)) for f in files]


def build_summary_tables(data):
    tables = {}
    p = data["patient"]
    if not p.empty:
        tables["Total Patients"] = p["Id"].nunique() if "Id" in p.columns else len(p)
        if "Gender" in p.columns:
            tables["Gender Split"] = p["Gender"].value_counts().to_dict()
        if "Age" in p.columns:
            tables["Average Age"] = round(p["Age"].mean(), 1)
    c = data["consultations"]
    if not c.empty and "Diagnosis" in c.columns:
        tables["Top 5 Diagnoses"] = c["Diagnosis"].value_counts().head(5).to_dict()
    ph = data["pharmacy"]
    if not ph.empty and "Drug Name" in ph.columns:
        tables["Top 5 Drugs"] = ph["Drug Name"].value_counts().head(5).to_dict()
    return tables


HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8"/>
<title>COREP Analysis Report</title>
<style>
  body { font-family: 'Segoe UI', Arial, sans-serif; margin: 40px;
         background: #f7f9fb; color: #223; }
  h1 { color: #0b4a6f; border-bottom: 3px solid #0b4a6f; padding-bottom: 6px; }
  h2 { color: #0b4a6f; margin-top: 40px; }
  .meta { color: #666; font-size: 0.9em; margin-bottom: 20px; }
  .card { background: #fff; padding: 18px 22px; border-radius: 10px;
          box-shadow: 0 1px 4px rgba(0,0,0,.08); margin: 14px 0; }
  table { border-collapse: collapse; width: 60%; margin: 10px 0; }
  th, td { padding: 8px 12px; border-bottom: 1px solid #eee; text-align: left; }
  th { background: #0b4a6f; color: #fff; }
  .chart { background: #fff; padding: 14px; border-radius: 10px;
           box-shadow: 0 1px 4px rgba(0,0,0,.08); margin: 18px 0; text-align:center; }
  .chart img { max-width: 100%; height: auto; border-radius: 6px; }
  .caption { color: #555; font-size: 0.9em; margin-top: 8px; }
</style>
</head>
<body>
  <h1>COREP Clinical Data Analysis Report</h1>
  <div class="meta">Generated: {{ generated_at }}</div>

  <h2>1. Executive Summary</h2>
  <div class="card">
    <table>
      {% for k, v in summary.items() %}
      <tr><th>{{ k }}</th><td>{{ v }}</td></tr>
      {% endfor %}
    </table>
  </div>

  <h2>2. Charts & Findings</h2>
  {% for name, b64 in charts %}
  <div class="chart">
    <img src="data:image/png;base64,{{ b64 }}" alt="{{ name }}"/>
    <div class="caption">{{ name.replace('_',' ').replace('.png','').title() }}</div>
  </div>
  {% endfor %}

  <h2>3. Key Insights</h2>
  <div class="card">
    <ul>
      <li>Patient registrations show a consistent trend - monitor monthly volume for resource planning.</li>
      <li>Top diagnoses drive most of the clinical workload - target training &amp; drug stocking accordingly.</li>
      <li>Referral pathway indicates the strongest downstream load (Lab vs Pharmacy vs Optical).</li>
      <li>Low-stock alerts must be actioned promptly to avoid stockouts.</li>
    </ul>
  </div>
</body>
</html>
"""


def generate_report():
    print("\n=== GENERATING REPORT ===")
    data = analysis.run()

    summary = build_summary_tables(data)
    charts = [(name, img_to_base64(path)) for name, path in collect_charts()]

    html = Template(HTML_TEMPLATE).render(
        generated_at=datetime.now().strftime("%Y-%m-%d %H:%M"),
        summary=summary,
        charts=charts,
    )
    out_html = os.path.join(REPORT_DIR, "corep_report.html")
    with open(out_html, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"📄 HTML report → {out_html}")

    # Optional PDF
    try:
        from weasyprint import HTML
        out_pdf = os.path.join(REPORT_DIR, "corep_report.pdf")
        HTML(string=html).write_pdf(out_pdf)
        print(f"📕 PDF report  → {out_pdf}")
    except Exception as e:
        print(f"ℹ️  PDF skipped (weasyprint not installed or failed): {e}")

    return out_html


if __name__ == "__main__":
    generate_report()