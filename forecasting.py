# Time-Series Forecasting (Prophet + ARIMA)


# """
# Patient volume forecasting.
# - Forecasts DAILY patient registrations and consultations.
# - Uses Prophet (if available) and statsmodels ARIMA as fallback.
# - Saves plots to ./charts/ and forecast CSV to ./reports/.
# """

# import os
# import warnings
# warnings.filterwarnings("ignore")

# import numpy as np
# import pandas as pd
# import matplotlib
# matplotlib.use("Agg")
# import matplotlib.pyplot as plt

# from statsmodels.tsa.arima.model import ARIMA
# from statsmodels.tsa.stattools import adfuller

# CHART_DIR = "charts"
# REPORT_DIR = "reports"
# os.makedirs(CHART_DIR, exist_ok=True)
# os.makedirs(REPORT_DIR, exist_ok=True)

# FORECAST_DAYS = 30


# def _save(fig, name):
#     path = os.path.join(CHART_DIR, name)
#     fig.savefig(path, dpi=110, bbox_inches="tight")
#     plt.close(fig)
#     print(f"  💾 {path}")


# def _daily_series(df, date_col="Created At"):
#     s = df.copy()
#     s[date_col] = pd.to_datetime(s[date_col], errors="coerce")
#     s = s.dropna(subset=[date_col])
#     daily = s.groupby(s[date_col].dt.date).size()
#     daily.index = pd.to_datetime(daily.index)
#     daily = daily.asfreq("D").fillna(0)
#     return daily


# # ---------------- ARIMA ----------------
# def arima_forecast(series, name, days=FORECAST_DAYS):
#     if len(series) < 30:
#         print(f"  ⚠️  Not enough data for ARIMA ({len(series)} points). Skipping {name}.")
#         return None

#     # Stationarity check
#     try:
#         p_val = adfuller(series)[1]
#         d = 0 if p_val < 0.05 else 1
#     except Exception:
#         d = 1

#     try:
#         model = ARIMA(series, order=(2, d, 2))
#         fit = model.fit()
#     except Exception:
#         model = ARIMA(series, order=(1, 1, 1))
#         fit = model.fit()

#     fc = fit.get_forecast(steps=days)
#     mean = fc.predicted_mean
#     ci = fc.conf_int()

#     # Plot
#     fig, ax = plt.subplots(figsize=(14, 5))
#     ax.plot(series.index, series.values, label="Historical", color="navy")
#     ax.plot(mean.index, mean.values, label="Forecast", color="crimson")
#     ax.fill_between(ci.index, ci.iloc[:, 0], ci.iloc[:, 1],
#                     color="crimson", alpha=0.2, label="95% CI")
#     ax.set_title(f"ARIMA Forecast – {name} ({days} days)")
#     ax.set_ylabel("Count"); ax.legend()
#     plt.tight_layout()
#     _save(fig, f"forecast_arima_{name}.png")

#     out = pd.DataFrame({
#         "date": mean.index,
#         "forecast": mean.values,
#         "lower": ci.iloc[:, 0].values,
#         "upper": ci.iloc[:, 1].values,
#     })
#     out.to_csv(os.path.join(REPORT_DIR, f"forecast_{name}_arima.csv"), index=False)
#     print(f"  ✅ ARIMA({2},{d},2) → next {days} days avg = {mean.mean():.1f}")
#     return out


# # ---------------- Prophet ----------------
# def prophet_forecast(series, name, days=FORECAST_DAYS):
#     try:
#         from prophet import Prophet
#     except ImportError:
#         print("  ℹ️  Prophet not installed – skipping.")
#         return None

#     df = pd.DataFrame({"ds": series.index, "y": series.values})

#     m = Prophet(
#         daily_seasonality=False,
#         weekly_seasonality=True,
#         yearly_seasonality=True,
#         changepoint_prior_scale=0.1,
#     )
#     m.fit(df)

#     future = m.make_future_dataframe(periods=days)
#     fc = m.predict(future)

#     fig = m.plot(fc)
#     plt.title(f"Prophet Forecast – {name} ({days} days)")
#     plt.tight_layout()
#     _save(fig, f"forecast_prophet_{name}.png")

#     fig2 = m.plot_components(fc)
#     plt.tight_layout()
#     _save(fig2, f"forecast_prophet_components_{name}.png")

#     out = fc[["ds", "yhat", "yhat_lower", "yhat_upper"]].tail(days)
#     out.to_csv(os.path.join(REPORT_DIR, f"forecast_{name}_prophet.csv"), index=False)
#     print(f"  ✅ Prophet → next {days} days avg = {out['yhat'].mean():.1f}")
#     return out


# def run(patient, consultations):
#     print("\n=== FORECASTING MODULE ===")

#     # Registrations
#     if not patient.empty and "Created At" in patient.columns:
#         reg = _daily_series(patient)
#         print(f"  Registrations: {len(reg)} days of data")
#         prophet_forecast(reg, "registrations")
#         arima_forecast(reg, "registrations")

#     # Consultations
#     if not consultations.empty and "Created At" in consultations.columns:
#         cons = _daily_series(consultations)
#         print(f"  Consultations: {len(cons)} days of data")
#         prophet_forecast(cons, "consultations")
#         arima_forecast(cons, "consultations")


# if __name__ == "__main__":
#     import main_analysis as analysis
#     d = analysis.clean_all(analysis.load_data())
#     run(d["patient"], d["consultations"])


"""
Patient volume forecasting — ADAPTIVE
Detects the data span and chooses the right strategy:

  • Multi-day data (≥14 days)  → Prophet + ARIMA daily forecast
  • Single-day outreach         → Hourly arrival forecast (logistics planning)
  • No usable dates             → Graceful skip

Outputs:
  ./charts/forecast_*.png
  ./reports/forecast_*.csv
"""

import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CHART_DIR = "charts"
REPORT_DIR = "reports"
os.makedirs(CHART_DIR, exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)

FORECAST_DAYS = 30
MIN_DAYS_FOR_PROPHET = 14   # need at least this many distinct days
MIN_DAYS_FOR_ARIMA   = 30


def _save(fig, name):
    path = os.path.join(CHART_DIR, name)
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"  💾 {path}")


# ============================================================
# Build a daily series from a date column
# ============================================================
def _daily_series(df, date_col="Created At"):
    if df is None or df.empty or date_col not in df.columns:
        return pd.Series(dtype=float)
    s = df.copy()
    s[date_col] = pd.to_datetime(s[date_col], errors="coerce")
    s = s.dropna(subset=[date_col])
    if s.empty:
        return pd.Series(dtype=float)
    daily = s.groupby(s[date_col].dt.date).size()
    daily.index = pd.to_datetime(daily.index)
    daily = daily.asfreq("D").fillna(0)
    return daily


# ============================================================
# Hourly series (for single-day outreach)
# ============================================================
def _hourly_series(df, date_col="Created At"):
    if df is None or df.empty or date_col not in df.columns:
        return pd.Series(dtype=float)
    s = df.copy()
    s[date_col] = pd.to_datetime(s[date_col], errors="coerce")
    s = s.dropna(subset=[date_col])
    if s.empty:
        return pd.Series(dtype=float)
    # Group by hour of day (0–23)
    hourly = s.groupby(s[date_col].dt.hour).size()
    hourly = hourly.reindex(range(24), fill_value=0)
    return hourly


# ============================================================
# Prophet (multi-day only)
# ============================================================
def prophet_forecast(series, name, days=FORECAST_DAYS):
    try:
        from prophet import Prophet
    except ImportError:
        print("  ℹ️  Prophet not installed — skipping.")
        return None

    if series is None or len(series) < 2:
        print(f"  ⚠️  Not enough data for Prophet ({len(series)} points). Skipping {name}.")
        return None

    df = pd.DataFrame({"ds": series.index, "y": series.values}).dropna()
    if len(df) < 2:
        print(f"  ⚠️  Prophet needs ≥2 non-NaN rows. Skipping {name}.")
        return None

    try:
        m = Prophet(
            daily_seasonality=False,
            weekly_seasonality=True,
            yearly_seasonality=False,   # not enough data for yearly
            changepoint_prior_scale=0.1,
        )
        m.fit(df)

        future = m.make_future_dataframe(periods=days)
        fc = m.predict(future)

        fig = m.plot(fc)
        plt.title(f"Prophet Forecast – {name} ({days} days)")
        plt.tight_layout()
        _save(fig, f"forecast_prophet_{name}.png")

        out = fc[["ds", "yhat", "yhat_lower", "yhat_upper"]].tail(days)
        out.to_csv(os.path.join(REPORT_DIR, f"forecast_{name}_prophet.csv"), index=False)
        print(f"  ✅ Prophet → next {days} days avg = {out['yhat'].mean():.1f}")
        return out
    except Exception as e:
        print(f"  ⚠️  Prophet failed for {name}: {e}")
        return None


# ============================================================
# ARIMA (multi-day only)
# ============================================================
def arima_forecast(series, name, days=FORECAST_DAYS):
    if series is None or len(series) < MIN_DAYS_FOR_ARIMA:
        print(f"  ⚠️  Not enough data for ARIMA ({len(series)} points, need {MIN_DAYS_FOR_ARIMA}). Skipping {name}.")
        return None

    from statsmodels.tsa.arima.model import ARIMA
    from statsmodels.tsa.stattools import adfuller

    try:
        p_val = adfuller(series.dropna())[1]
        d = 0 if p_val < 0.05 else 1
    except Exception:
        d = 1

    try:
        model = ARIMA(series, order=(2, d, 2))
        fit = model.fit()
    except Exception:
        try:
            model = ARIMA(series, order=(1, 1, 1))
            fit = model.fit()
        except Exception as e:
            print(f"  ⚠️  ARIMA failed for {name}: {e}")
            return None

    fc = fit.get_forecast(steps=days)
    mean = fc.predicted_mean
    ci = fc.conf_int()

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
    print(f"  ✅ ARIMA(2,{d},2) → next {days} days avg = {mean.mean():.1f}")
    return out


# ============================================================
# Intraday forecast for single-day outreach
# ============================================================
def intraday_forecast(df, name):
    """
    For single-day outreach:
      - Build hourly arrival counts
      - Smooth with a rolling average
      - Report peak hours + estimated staffing needs
    """
    if df is None or df.empty or "Created At" not in df.columns:
        print(f"  ⚠️  No dates for intraday analysis ({name}).")
        return None

    hourly = _hourly_series(df, "Created At")
    if hourly.empty or hourly.sum() == 0:
        print(f"  ⚠️  No usable hours in data ({name}).")
        return None

    # Rolling average (window 3) for smoother planning curve
    smooth = hourly.rolling(window=3, center=True, min_periods=1).mean()

    # Find peak hour
    peak_hour = int(hourly.idxmax())
    peak_val = int(hourly.max())
    total = int(hourly.sum())

    # Simple staffing estimate: 1 staff per 5 arrivals/hour
    staff_peak = max(1, peak_val // 5)

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.bar(hourly.index, hourly.values, color="#1a7fad",
           edgecolor="black", alpha=0.7, label="Arrivals/hour")
    ax.plot(smooth.index, smooth.values, color="#e63946",
            linewidth=3, marker="o", label="3-hour rolling avg")
    ax.axvline(peak_hour, color="green", linestyle="--",
               linewidth=2, label=f"Peak @ {peak_hour}:00 ({peak_val})")
    ax.set_xlabel("Hour of Day", fontsize=12, fontweight="bold")
    ax.set_ylabel("Arrivals", fontsize=12, fontweight="bold")
    ax.set_title(f"Intraday {name.title()} Forecast — Single-Day Outreach",
                 fontsize=16, fontweight="bold", color="#0b4a6f")
    ax.set_xticks(range(24))
    ax.legend()
    plt.tight_layout()
    _save(fig, f"forecast_intraday_{name}.png")

    out = pd.DataFrame({
        "hour": hourly.index,
        "arrivals": hourly.values,
        "rolling_avg": smooth.values.round(2),
    })
    out.to_csv(os.path.join(REPORT_DIR, f"forecast_intraday_{name}.csv"), index=False)

    print(f"  ✅ Intraday {name}: peak at {peak_hour}:00 ({peak_val} arrivals), "
          f"suggest ~{staff_peak} staff at peak, total {total}.")
    return out


# ============================================================
# Main dispatcher
# ============================================================
def run(patient, consultations):
    print("\n=== FORECASTING MODULE ===")

    # ---- Registrations ----
    if not patient.empty and "Created At" in patient.columns:
        reg_daily = _daily_series(patient, "Created At")
        reg_days = reg_daily[reg_daily > 0].count()
        print(f"  Registrations: {reg_days} active day(s), {len(reg_daily)} calendar days")

        if reg_days >= MIN_DAYS_FOR_PROPHET:
            # Multi-day: full time-series forecast
            prophet_forecast(reg_daily, "registrations")
            arima_forecast(reg_daily, "registrations")
        else:
            # Single-day: intraday forecast
            print("  ℹ️  Single-day data → switching to intraday forecast")
            intraday_forecast(patient, "registrations")

    # ---- Consultations ----
    if not consultations.empty and "Created At" in consultations.columns:
        cons_daily = _daily_series(consultations, "Created At")
        cons_days = cons_daily[cons_daily > 0].count()
        print(f"  Consultations: {cons_days} active day(s), {len(cons_daily)} calendar days")

        if cons_days >= MIN_DAYS_FOR_PROPHET:
            prophet_forecast(cons_daily, "consultations")
            arima_forecast(cons_daily, "consultations")
        else:
            print("  ℹ️  Single-day data → switching to intraday forecast")
            intraday_forecast(consultations, "consultations")

    print("\n✅ Forecasting module complete.")


if __name__ == "__main__":
    import main_analysis as analysis
    d = analysis.clean_all(analysis.load_data())
    run(d["patient"], d["consultations"])