"""
Predictive Forecasting of Care Load & Placement Demand
Simple, easy-to-follow forecasting pipeline.

What this script does:
1. Loads the engineered dataset
2. Splits data by time (train / validation / test)
3. Trains baseline and statistical models (+ simple linear ML)
4. Forecasts HHS Care load and Discharge demand
5. Saves metrics, forecasts, and KPI summary to the outputs folder

Dependencies used: numpy, pandas, statsmodels only (no scikit-learn).
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from datetime import timedelta
from pathlib import Path

from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing

# -------------------------------------------------
# Paths
# -------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "final_dataset.csv"
OUTPUT_DIR = BASE_DIR / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

# -------------------------------------------------
# 1. Load data
# -------------------------------------------------
df = pd.read_csv(DATA_PATH)
df["Date"] = pd.to_datetime(df["Date"])
df = df.sort_values("Date").reset_index(drop=True)

# Drop early rows that still have NaN lags (needed for feature-based models)
df_ml = df.dropna().reset_index(drop=True)

print("Data loaded.")
print("Date range:", df["Date"].min().date(), "to", df["Date"].max().date())
print("Total days:", len(df))
print("Days usable for feature models (after lag NaNs removed):", len(df_ml))

# -------------------------------------------------
# 2. Time-based train / validation / test split
#    (~67% train, ~25% val, rest test) — same idea as the notebook
# -------------------------------------------------
initial_day = df["Date"].iloc[0]
final_day = df["Date"].iloc[-1]
total_days = (final_day - initial_day).days

n_train = round(total_days * 0.67)
n_val = round(total_days * 0.25)

train_end = initial_day + timedelta(days=n_train - 1)
val_end = train_end + timedelta(days=n_val - 1)

train_df = df[df["Date"] <= train_end].copy()
val_df = df[(df["Date"] > train_end) & (df["Date"] <= val_end)].copy()
test_df = df[df["Date"] > val_end].copy()

print("\nSplit sizes:")
print("  Train:", len(train_df), "| Val:", len(val_df), "| Test:", len(test_df))

train_ml = df_ml[df_ml["Date"] <= train_end].copy()
test_ml = df_ml[df_ml["Date"] > val_end].copy()

# -------------------------------------------------
# Helper: evaluation metrics
# -------------------------------------------------
def calc_metrics(actual, predicted):
    """Return MAE, RMSE, MAPE for one series of forecasts."""
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    mae = np.mean(np.abs(actual - predicted))
    rmse = np.sqrt(np.mean((actual - predicted) ** 2))
    nonzero = actual != 0
    if nonzero.sum() == 0:
        mape = np.nan
    else:
        mape = np.mean(np.abs((actual[nonzero] - predicted[nonzero]) / actual[nonzero])) * 100
    return round(float(mae), 2), round(float(rmse), 2), round(float(mape), 2)


def accuracy_pct(mape):
    """Simple forecast accuracy % from MAPE."""
    if pd.isna(mape):
        return np.nan
    return round(max(0, 100 - mape), 2)


# -------------------------------------------------
# 3. Baseline models
# -------------------------------------------------
def naive_forecast(series, horizon):
    """Repeat the last observed value for every future day."""
    last_value = float(series.iloc[-1])
    return np.full(horizon, last_value)


def moving_average_forecast(series, horizon, window=7):
    """Use the recent rolling mean as the constant forecast."""
    avg = float(series.tail(window).mean())
    return np.full(horizon, avg)


# -------------------------------------------------
# 4. ARIMA
# -------------------------------------------------
def arima_forecast(series, horizon, order=(2, 1, 2)):
    """Fit ARIMA and forecast horizon steps ahead."""
    try:
        model = ARIMA(series.astype(float), order=order)
        fitted = model.fit()
        pred = fitted.forecast(steps=horizon)
        return np.asarray(pred, dtype=float)
    except Exception as e:
        print("  ARIMA failed:", e)
        return naive_forecast(series, horizon)


# -------------------------------------------------
# 5. Exponential Smoothing (Holt linear trend)
# -------------------------------------------------
def exp_smoothing_forecast(series, horizon):
    """Holt linear trend (no strong weekly seasonality assumed)."""
    try:
        model = ExponentialSmoothing(
            series.astype(float),
            trend="add",
            seasonal=None,
            initialization_method="estimated"
        )
        fitted = model.fit(optimized=True)
        pred = fitted.forecast(horizon)
        return np.asarray(pred, dtype=float)
    except Exception as e:
        print("  ExpSmoothing failed:", e)
        return naive_forecast(series, horizon)


# -------------------------------------------------
# 6. Simple linear "ML" model (numpy least squares)
#    This stands in for Random Forest / Gradient Boosting
#    when scikit-learn is not available. Same features are used.
# -------------------------------------------------
FEATURE_COLS = [
    "Placed_In_CBP_lag_1", "CBP_Custody_lag_1", "CBP_To_HHS_lag_1",
    "HHS_Care_lag_1", "Discharged_From_HHS_lag_1",
    "Placed_In_CBP_lag_7", "CBP_Custody_lag_7", "CBP_To_HHS_lag_7",
    "HHS_Care_lag_7", "Discharged_From_HHS_lag_7",
    "Placed_In_CBP_lag_14", "CBP_Custody_lag_14", "CBP_To_HHS_lag_14",
    "HHS_Care_lag_14", "Discharged_From_HHS_lag_14",
    "HHS_Care_rolling_avg_7", "Discharged_From_HHS_rolling_avg_7",
    "HHS_Care_rolling_avg_14", "Discharged_From_HHS_rolling_avg_14",
    "Net_HHS_Flow", "Net_HHS_Flow_rolling_avg_7",
    "Day_Of_Week", "Month", "Is_Weekend"
]


def train_linear_model(train_frame, target):
    """
    Ordinary least-squares linear regression with numpy.
    Returns coefficient vector (including intercept).
    """
    X = train_frame[FEATURE_COLS].astype(float).values
    y = train_frame[target].astype(float).values
    # Add intercept column
    X_design = np.column_stack([np.ones(len(X)), X])
    # Solve least squares
    coeffs, residuals, rank, s = np.linalg.lstsq(X_design, y, rcond=None)
    return coeffs


def linear_predict(coeffs, frame):
    """Predict with the linear model coefficients."""
    X = frame[FEATURE_COLS].astype(float).values
    X_design = np.column_stack([np.ones(len(X)), X])
    return X_design @ coeffs


# -------------------------------------------------
# 7. Evaluate models on the TEST set
# -------------------------------------------------
TARGETS = ["HHS_Care", "Discharged_From_HHS"]
HORIZON = len(test_df)

results = []
forecast_store = {}

for target in TARGETS:
    print("\n" + "=" * 60)
    print("Target:", target)
    print("=" * 60)

    train_series = train_df.set_index("Date")[target]
    actual_test = test_df[target].values.astype(float)
    dates_test = test_df["Date"].values

    # Naive
    pred = naive_forecast(train_series, HORIZON)
    mae, rmse, mape = calc_metrics(actual_test, pred)
    results.append({"Target": target, "Model": "Naive", "MAE": mae, "RMSE": rmse,
                    "MAPE": mape, "Forecast_Accuracy_Pct": accuracy_pct(mape)})
    forecast_store[(target, "Naive")] = pred
    print(f"  Naive            MAE={mae:8.2f}  RMSE={rmse:8.2f}  MAPE={mape:6.2f}%")

    # Moving Average
    pred = moving_average_forecast(train_series, HORIZON, window=7)
    mae, rmse, mape = calc_metrics(actual_test, pred)
    results.append({"Target": target, "Model": "Moving_Average", "MAE": mae, "RMSE": rmse,
                    "MAPE": mape, "Forecast_Accuracy_Pct": accuracy_pct(mape)})
    forecast_store[(target, "Moving_Average")] = pred
    print(f"  Moving Average   MAE={mae:8.2f}  RMSE={rmse:8.2f}  MAPE={mape:6.2f}%")

    # ARIMA
    pred = arima_forecast(train_series, HORIZON, order=(2, 1, 2))
    mae, rmse, mape = calc_metrics(actual_test, pred)
    results.append({"Target": target, "Model": "ARIMA", "MAE": mae, "RMSE": rmse,
                    "MAPE": mape, "Forecast_Accuracy_Pct": accuracy_pct(mape)})
    forecast_store[(target, "ARIMA")] = pred
    print(f"  ARIMA            MAE={mae:8.2f}  RMSE={rmse:8.2f}  MAPE={mape:6.2f}%")

    # Exponential Smoothing
    pred = exp_smoothing_forecast(train_series, HORIZON)
    mae, rmse, mape = calc_metrics(actual_test, pred)
    results.append({"Target": target, "Model": "ExpSmoothing", "MAE": mae, "RMSE": rmse,
                    "MAPE": mape, "Forecast_Accuracy_Pct": accuracy_pct(mape)})
    forecast_store[(target, "ExpSmoothing")] = pred
    print(f"  ExpSmoothing     MAE={mae:8.2f}  RMSE={rmse:8.2f}  MAPE={mape:6.2f}%")

    # Linear feature model (proxy for RF / Gradient Boosting)
    coeffs = train_linear_model(train_ml, target)
    pred_ml = linear_predict(coeffs, test_ml)
    actual_ml = test_ml[target].values.astype(float)
    mae, rmse, mape = calc_metrics(actual_ml, pred_ml)
    results.append({"Target": target, "Model": "LinearFeatures", "MAE": mae, "RMSE": rmse,
                    "MAPE": mape, "Forecast_Accuracy_Pct": accuracy_pct(mape)})
    forecast_store[(target, "LinearFeatures")] = pred_ml
    print(f"  LinearFeatures   MAE={mae:8.2f}  RMSE={rmse:8.2f}  MAPE={mape:6.2f}%")

# -------------------------------------------------
# 8. Save metric table
# -------------------------------------------------
metrics_df = pd.DataFrame(results)
metrics_path = OUTPUT_DIR / "model_metrics.csv"
metrics_df.to_csv(metrics_path, index=False)
print("\nSaved metrics to", metrics_path)

# -------------------------------------------------
# 9. 14-day forward forecast from the last observed day
# -------------------------------------------------
FORWARD_DAYS = 14
last_date = df["Date"].max()
future_dates = pd.date_range(last_date + timedelta(days=1), periods=FORWARD_DAYS, freq="D")

forward_rows = []

for target in TARGETS:
    full_series = df.set_index("Date")[target]

    fwd_naive = naive_forecast(full_series, FORWARD_DAYS)
    fwd_ma = moving_average_forecast(full_series, FORWARD_DAYS, window=7)
    fwd_arima = arima_forecast(full_series, FORWARD_DAYS, order=(2, 1, 2))
    fwd_es = exp_smoothing_forecast(full_series, FORWARD_DAYS)

    # Linear model on all history; hold last feature row constant
    coeffs_full = train_linear_model(df_ml, target)
    last_feat = df_ml[FEATURE_COLS].iloc[[-1]]
    future_feat = pd.concat([last_feat] * FORWARD_DAYS, ignore_index=True)
    fwd_lin = linear_predict(coeffs_full, future_feat)

    for i, d in enumerate(future_dates):
        forward_rows.append({
            "Date": d,
            "Target": target,
            "Naive": round(float(fwd_naive[i]), 1),
            "Moving_Average": round(float(fwd_ma[i]), 1),
            "ARIMA": round(float(fwd_arima[i]), 1),
            "ExpSmoothing": round(float(fwd_es[i]), 1),
            "LinearFeatures": round(float(fwd_lin[i]), 1),
        })

forward_df = pd.DataFrame(forward_rows)
forward_path = OUTPUT_DIR / "forward_forecast_14day.csv"
forward_df.to_csv(forward_path, index=False)
print("Saved 14-day forward forecast to", forward_path)

# -------------------------------------------------
# 10. Net pressure early warning (Transfers - Discharges)
# -------------------------------------------------
transfer_series = df.set_index("Date")["CBP_To_HHS"]
discharge_series = df.set_index("Date")["Discharged_From_HHS"]

fwd_transfers = arima_forecast(transfer_series, FORWARD_DAYS, order=(2, 1, 2))
fwd_discharges = arima_forecast(discharge_series, FORWARD_DAYS, order=(2, 1, 2))
net_pressure = fwd_transfers - fwd_discharges

pressure_df = pd.DataFrame({
    "Date": future_dates,
    "Forecast_Transfers_In": np.round(fwd_transfers, 1),
    "Forecast_Discharges": np.round(fwd_discharges, 1),
    "Net_Pressure": np.round(net_pressure, 1),
    "Capacity_Stress_Flag": np.where(
        net_pressure > 0, "Rising care load risk", "Discharge keeping pace"
    )
})
pressure_path = OUTPUT_DIR / "net_pressure_early_warning.csv"
pressure_df.to_csv(pressure_path, index=False)
print("Saved net pressure early-warning table to", pressure_path)

# -------------------------------------------------
# 11. KPI summary (best model per target by MAE)
# -------------------------------------------------
kpi_rows = []
for target in TARGETS:
    sub = metrics_df[metrics_df["Target"] == target]
    best = sub.loc[sub["MAE"].idxmin()]
    kpi_rows.append({
        "Target": target,
        "Best_Model": best["Model"],
        "Best_MAE": best["MAE"],
        "Best_RMSE": best["RMSE"],
        "Best_MAPE": best["MAPE"],
        "Forecast_Accuracy_Pct": best["Forecast_Accuracy_Pct"],
        "Surge_Lead_Time_Days": FORWARD_DAYS,
        "Notes": "Lower MAE/RMSE is better. Accuracy % = 100 - MAPE."
    })

kpi_df = pd.DataFrame(kpi_rows)
kpi_path = OUTPUT_DIR / "kpi_summary.csv"
kpi_df.to_csv(kpi_path, index=False)
print("Saved KPI summary to", kpi_path)

# -------------------------------------------------
# 12. Test-period predictions for charts
# -------------------------------------------------
test_pred_rows = []
for target in TARGETS:
    for model_name in ["Naive", "Moving_Average", "ARIMA", "ExpSmoothing", "LinearFeatures"]:
        preds = forecast_store.get((target, model_name))
        if preds is None:
            continue
        if model_name == "LinearFeatures":
            dates = test_ml["Date"].values
            actual = test_ml[target].values
        else:
            dates = test_df["Date"].values
            actual = test_df[target].values
        n = min(len(dates), len(preds), len(actual))
        for i in range(n):
            test_pred_rows.append({
                "Date": dates[i],
                "Target": target,
                "Model": model_name,
                "Actual": round(float(actual[i]), 1),
                "Predicted": round(float(preds[i]), 1)
            })

test_pred_df = pd.DataFrame(test_pred_rows)
test_pred_path = OUTPUT_DIR / "test_predictions.csv"
test_pred_df.to_csv(test_pred_path, index=False)
print("Saved test predictions to", test_pred_path)

print("\nDone. All outputs are in the 'outputs' folder.")
print("Next step: run the Streamlit app (app/streamlit_app.py).")
