# Ready for Multiple Years Forecasting


"""
Annual outreach forecasting.
Works with 1 year (baseline stats) → scales to multi-year (trend + forecast).
"""

import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

CHART_DIR = "charts"
REPORT_DIR = "reports"
os.makedirs(CHART_DIR, exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)


def _save(fig, name):
    path = os.path.join(CHART_DIR, name)
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"  💾 {path}")


def _yearly_counts(df, date_col="Created At"):
    """Aggregate by calendar year."""
    if df is None or df.empty or date_col not in df.columns:
        return pd.Series(dtype=int)
    s = df.copy()
    s[date_col] = pd.to_datetime(s[date_col], errors="coerce")
    s = s.dropna(subset=[date_col])
    if s.empty:
        return pd.Series(dtype=int)
    yearly = s.groupby(s[date_col].dt.year).size()
    yearly.index.name = "Year"
    return yearly


def annual_baseline(df, label):
    """Compute and visualize annual stats for one table."""
    if df is None or df.empty:
        print(f"  ⚠️  No data for {label}")
        return None

    yearly = _yearly_counts(df)
    if yearly.empty:
        print(f"  ⚠️  No dates in {label}")
        return None

    print(f"\n  📅 {label.title()} by year:")
    for y, c in yearly.items():
        print(f"     • {y}: {c}")

    # Bar chart
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.barplot(x=yearly.index.astype(str), y=yearly.values,
                palette="viridis", ax=ax, edgecolor="black")
    for i, v in enumerate(yearly.values):
        ax.text(i, v + 0.5, str(v), ha="center", fontweight="bold")
    ax.set_title(f"{label.title()} — Annual Totals",
                 fontsize=16, fontweight="bold", color="#0b4a6f")
    ax.set_xlabel("Year", fontweight="bold")
    ax.set_ylabel("Count", fontweight="bold")
    plt.tight_layout()
    _save(fig, f"annual_{label.lower().replace(' ', '_')}.png")

    yearly.to_csv(os.path.join(REPORT_DIR, f"annual_{label.lower().replace(' ', '_')}.csv"))

    # Forecast if 2+ years
    if len(yearly) >= 2:
        _forecast_annual(yearly, label)
    else:
        print(f"  ℹ️  Only 1 year of data — forecast will activate with 2+ years.")

    return yearly


def _forecast_annual(yearly, label):
    """Simple linear trend forecast for next year."""
    years = yearly.index.values
    counts = yearly.values

    # Linear regression
    coef = np.polyfit(years, counts, 1)
    poly = np.poly1d(coef)

    next_year = years.max() + 1
    prediction = max(0, int(round(poly(next_year))))

    print(f"\n  🔮 {label.title()} forecast for {next_year}: {prediction}")
    print(f"     Trend: {'+' if coef[0] > 0 else ''}{coef[0]:.1f} per year")

    # Plot with trend
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.scatter(years, counts, s=120, color="#1a7fad",
               edgecolor="black", zorder=3, label="Actual")
    trend_years = np.arange(years.min(), next_year + 1)
    ax.plot(trend_years, poly(trend_years), "--", color="#e63946",
            linewidth=2, label="Trend")
    ax.scatter([next_year], [prediction], s=200, color="#2a9d8f",
               edgecolor="black", marker="*", zorder=4,
               label=f"Forecast {next_year}")
    ax.annotate(f"  {prediction}", (next_year, prediction),
                fontsize=14, fontweight="bold", color="#2a9d8f")
    ax.set_title(f"{label.title()} — Annual Forecast",
                 fontsize=16, fontweight="bold", color="#0b4a6f")
    ax.set_xlabel("Year", fontweight="bold")
    ax.set_ylabel("Count", fontweight="bold")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    _save(fig, f"annual_forecast_{label.lower().replace(' ', '_')}.png")


def run(data):
    print("\n" + "=" * 60)
    print(" ANNUAL OUTREACH ANALYSIS")
    print("=" * 60)

    print("\nℹ️  Outreach cadence: annual (1 event per year)")
    print("   This module tracks year-over-year growth and predicts future turnout.")

    annual_baseline(data.get("patient"), "patients")
    annual_baseline(data.get("consultations"), "consultations")
    annual_baseline(data.get("pharmacy"), "pharmacy orders")
    annual_baseline(data.get("lab_tests"), "lab tests")

    print("\n✅ Annual analysis complete.")
    print("   → Charts: charts/annual_*.png")
    print("   → CSVs:   reports/annual_*.csv")


if __name__ == "__main__":
    import main_analysis as analysis
    d = analysis.clean_all(analysis.load_data())
    run(d)