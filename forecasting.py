# Time-Series Forecasting (Prophet + ARIMA)


"""
Patient volume forecasting.
- Forecasts DAILY patient registrations and consultations.
- Uses Prophet (if available) and statsmodels ARIMA as fallback.
- Saves plots to ./charts/ and forecast CSV to ./reports/.
"""

import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller

CHART_DIR = "charts"
REPORT_DIR = "reports"
os.makedirs(CHART_DIR, exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)

FORECAST_DAYS = 30


def _save(fig, name):
    path = os.path.join(CHART_DIR, name)
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"  💾 {path}")


def _daily_series(df, date_col="Created At"):
    s = df.copy()
    s[date_col] = pd.to_datetime(s[date_col], errors="coerce")
    s = s.dropna(subset=[date_col])
    daily = s.groupby(s[date_col].dt.date).size()
    daily.index = pd.to_datetime(daily.index)
    daily = daily.asfreq("D").fillna(0)
    return daily


# ---------------- ARIMA ----------------
def arima_forecast(series, name, days=FORECAST_DAYS):
    if len(series) < 30:
        print(f"  ⚠️  Not enough data for ARIMA ({len(series)} points). Skipping {name}.")
        return None

    # Stationarity check
    try:
        p_val = adfuller(series)[1]
        d = 0 if p_val < 0.05 else 1
    except Exception:
        d = 1

    try:
        model = ARIMA(series, order=(2, d, 2))
        fit = model.fit()
    except Exception:
        model = ARIMA(series, order=(1, 1, 1))
        fit = model.fit()

    fc = fit.get_forecast(steps=days)
    mean = fc.predicted_mean
    ci = fc.conf_int()

    # Plot
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(series.index, series.values, label="Historical", color="navy")
    ax.plot(mean.index, mean.values, label="Forecast", color="crimson")
    ax.fill_between(ci.index, ci.iloc[:, 0], ci.iloc[:, 1],
                    color="crimson", alpha=0.2, label="95% CI")
    ax.set_title(f"ARIMA Forecast – {name} ({days} days)")
    ax.set_ylabel("Count"); ax.legend()
    plt.tight_layout()
    _save(fig, f"forecast_arima_{name}.png")

    out = pd.DataFrame({
        "date": mean.index,
        "forecast": mean.values,
        "lower": ci.iloc[:, 0].values,
        "upper": ci.iloc[:, 1].values,
    })
    out.to_csv(os.path.join(REPORT_DIR, f"forecast_{name}_arima.csv"), index=False)
    print(f"  ✅ ARIMA({2},{d},2) → next {days} days avg = {mean.mean():.1f}")
    return out


# ---------------- Prophet ----------------
def prophet_forecast(series, name, days=FORECAST_DAYS):
    try:
        from prophet import Prophet
    except ImportError:
        print("  ℹ️  Prophet not installed – skipping.")
        return None

    df = pd.DataFrame({"ds": series.index, "y": series.values})

    m = Prophet(
        daily_seasonality=False,
        weekly_seasonality=True,
        yearly_seasonality=True,
        changepoint_prior_scale=0.1,
    )
    m.fit(df)

    future = m.make_future_dataframe(periods=days)
    fc = m.predict(future)

    fig = m.plot(fc)
    plt.title(f"Prophet Forecast – {name} ({days} days)")
    plt.tight_layout()
    _save(fig, f"forecast_prophet_{name}.png")

    fig2 = m.plot_components(fc)
    plt.tight_layout()
    _save(fig2, f"forecast_prophet_components_{name}.png")

    out = fc[["ds", "yhat", "yhat_lower", "yhat_upper"]].tail(days)
    out.to_csv(os.path.join(REPORT_DIR, f"forecast_{name}_prophet.csv"), index=False)
    print(f"  ✅ Prophet → next {days} days avg = {out['yhat'].mean():.1f}")
    return out


def run(patient, consultations):
    print("\n=== FORECASTING MODULE ===")

    # Registrations
    if not patient.empty and "Created At" in patient.columns:
        reg = _daily_series(patient)
        print(f"  Registrations: {len(reg)} days of data")
        prophet_forecast(reg, "registrations")
        arima_forecast(reg, "registrations")

    # Consultations
    if not consultations.empty and "Created At" in consultations.columns:
        cons = _daily_series(consultations)
        print(f"  Consultations: {len(cons)} days of data")
        prophet_forecast(cons, "consultations")
        arima_forecast(cons, "consultations")


if __name__ == "__main__":
    import main_analysis as analysis
    d = analysis.clean_all(analysis.load_data())
    run(d["patient"], d["consultations"])