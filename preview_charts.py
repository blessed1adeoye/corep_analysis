"""
Generates a single responsive HTML page displaying all charts from ./charts/
Run:  python preview_charts.py
Opens: reports/all_charts_preview.html
"""

import os
import base64
from datetime import datetime

CHART_DIR = "charts"
REPORT_DIR = "reports"
os.makedirs(REPORT_DIR, exist_ok=True)


def _b64(path):
    """Read PNG → base64 string."""
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def _fmt_size(bytes_):
    for unit in ["B", "KB", "MB"]:
        if bytes_ < 1024:
            return f"{bytes_:.1f} {unit}"
        bytes_ /= 1024
    return f"{bytes_:.1f} GB"


def _group(filename):
    """Group charts by prefix (01_, Q1_, forecast_, cluster_, nlp_, etc.)."""
    if filename.startswith("Q"):
        return "Clinical Questions"
    if filename.startswith("forecast_"):
        return "Forecasting"
    if filename.startswith("cluster_"):
        return "Clustering"
    if filename.startswith("nlp_"):
        return "NLP Analysis"
    if filename.startswith("ml_"):
        return "Machine Learning"
    return "Core EDA"


def main():
    if not os.path.isdir(CHART_DIR):
        print(f"❌ Chart folder not found: {CHART_DIR}")
        return

    files = sorted(f for f in os.listdir(CHART_DIR)
                   if f.lower().endswith(".png"))
    if not files:
        print(f"❌ No PNG files in {CHART_DIR}/")
        return

    # Group charts
    groups = {}
    for f in files:
        g = _group(f)
        groups.setdefault(g, []).append(f)

    # Build sections
    sections_html = []
    for group_name in ["Core EDA", "Clinical Questions", "Forecasting",
                       "Clustering", "NLP Analysis", "Machine Learning"]:
        if group_name not in groups:
            continue
        charts = groups[group_name]
        cards = []
        for f in charts:
            path = os.path.join(CHART_DIR, f)
            size = _fmt_size(os.path.getsize(path))
            mtime = datetime.fromtimestamp(os.path.getmtime(path)).strftime("%H:%M")
            cards.append(f"""
            <div class="chart-card">
              <div class="chart-header">
                <span class="chart-name">{f}</span>
                <span class="chart-meta">{size} · {mtime}</span>
              </div>
              <img src="data:image/png;base64,{_b64(path)}" alt="{f}" loading="lazy"/>
            </div>""")

        sections_html.append(f"""
        <section class="group">
          <h2>{group_name} <span class="badge">{len(charts)}</span></h2>
          <div class="grid">{''.join(cards)}</div>
        </section>""")

    total_size = sum(os.path.getsize(os.path.join(CHART_DIR, f)) for f in files)

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1.0"/>
<title>COREP — All Charts Preview</title>
<style>
  * {{ box-sizing: border-box; }}
  html, body {{
    margin: 0; padding: 0;
    font-family: 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
    background: #f7f9fb;
    color: #223;
    line-height: 1.55;
  }}
  header {{
    background: linear-gradient(135deg, #0b4a6f 0%, #1a7fad 100%);
    color: #fff;
    padding: 28px 32px;
    box-shadow: 0 4px 14px rgba(0,0,0,0.08);
  }}
  header h1 {{ margin: 0; font-size: 1.7rem; font-weight: 700; }}
  header p {{ margin: 6px 0 0; opacity: 0.9; font-size: 0.95rem; }}
  .stats {{
    display: flex; gap: 24px; margin-top: 14px; flex-wrap: wrap;
  }}
  .stat {{
    background: rgba(255,255,255,0.15);
    padding: 8px 16px; border-radius: 8px;
    font-size: 0.9rem;
  }}
  .stat strong {{ font-size: 1.1rem; }}

  .container {{ max-width: 1600px; margin: 0 auto; padding: 24px; }}

  .group {{ margin-bottom: 48px; }}
  .group h2 {{
    color: #0b4a6f;
    font-size: 1.3rem;
    border-bottom: 3px solid #0b4a6f;
    padding-bottom: 8px;
    margin: 0 0 20px;
    display: flex;
    align-items: center;
    gap: 12px;
  }}
  .badge {{
    background: #1a7fad;
    color: #fff;
    font-size: 0.8rem;
    padding: 2px 10px;
    border-radius: 12px;
    font-weight: 500;
  }}

  .grid {{
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(420px, 1fr));
    gap: 20px;
  }}

  .chart-card {{
    background: #fff;
    border-radius: 10px;
    padding: 14px;
    box-shadow: 0 1px 4px rgba(0,0,0,0.08);
    transition: box-shadow 0.2s, transform 0.2s;
    overflow: hidden;
  }}
  .chart-card:hover {{
    box-shadow: 0 4px 16px rgba(0,0,0,0.12);
    transform: translateY(-2px);
  }}
  .chart-header {{
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 10px;
    padding-bottom: 8px;
    border-bottom: 1px solid #eef2f6;
    gap: 10px;
  }}
  .chart-name {{
    font-size: 0.8rem;
    font-weight: 600;
    color: #0b4a6f;
    word-break: break-all;
    flex: 1;
  }}
  .chart-meta {{
    font-size: 0.7rem;
    color: #8899aa;
    white-space: nowrap;
  }}
  .chart-card img {{
    width: 100%;
    height: auto;
    display: block;
    border-radius: 6px;
  }}

  footer {{
    text-align: center;
    padding: 32px;
    color: #8899aa;
    font-size: 0.85rem;
  }}

  @media (max-width: 480px) {{
    .grid {{ grid-template-columns: 1fr; }}
    header h1 {{ font-size: 1.3rem; }}
    .container {{ padding: 16px; }}
  }}
</style>
</head>
<body>
  <header>
    <h1>🏥 COREP — All Charts Preview</h1>
    <p>Complete visualization gallery from the COREP Analytics Pipeline</p>
    <div class="stats">
      <div class="stat"><strong>{len(files)}</strong> charts</div>
      <div class="stat"><strong>{len(groups)}</strong> categories</div>
      <div class="stat"><strong>{_fmt_size(total_size)}</strong> total</div>
      <div class="stat">Generated <strong>{datetime.now().strftime('%Y-%m-%d %H:%M')}</strong></div>
    </div>
  </header>

  <div class="container">
    {''.join(sections_html)}
  </div>

  <footer>
    COREP Analytics Pipeline · {datetime.now().strftime('%Y')}
  </footer>
</body>
</html>
"""

    out = os.path.join(REPORT_DIR, "all_charts_preview.html")
    with open(out, "w", encoding="utf-8") as f:
        f.write(html)

    print(f"\n✅ Preview generated: {out}")
    print(f"   📊 {len(files)} charts in {len(groups)} categories")
    print(f"   📦 {_fmt_size(total_size)} total size")
    print(f"\n   Open in browser:")
    print(f"   file:///{os.path.abspath(out).replace(os.sep, '/')}")
    print(f"\n   Or from terminal (Windows):")
    print(f"   start {out}\n")


if __name__ == "__main__":
    main()