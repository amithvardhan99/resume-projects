# Predictive Forecasting of Care Load & Placement Demand

## What this project does

Forecasts:

1. **Children in HHS Care** (care load)
2. **Discharges from HHS Care** (placement demand)
3. **Net pressure** (transfers in − discharges) as an early-warning capacity signal

## Already completed (your work)

- Data cleaning, interpolation, Tableau EDA (`Book1–3.twb`)
- Feature engineering → `final_dataset.csv`
- Notebook analysis for ARIMA, exponential smoothing, Random Forest, Gradient Boosting (`notebook.ipynb`, `src/`)

## Completed in this continuation

| Deliverable | Path |
|-------------|------|
| End-to-end forecasting script (simple code) | `complete_forecasting.py` |
| Model metrics, forecasts, KPIs | `outputs/*.csv` |
| Static dashboard (open in browser) | `outputs/dashboard.html` |
| Streamlit app | `app/streamlit_app.py` |
| Research paper | `Research_Paper_Care_Load_Forecasting.docx` |
| Executive summary | `Executive_Summary_Government_Stakeholders.docx` |

## How to run

```bash
# 1. Rebuild forecasts and metrics
python complete_forecasting.py

# 2a. Open the static dashboard
#     Double-click outputs/dashboard.html  (or open in any browser)

# 2b. Interactive Streamlit dashboard (needs: pip install streamlit plotly)
streamlit run app/streamlit_app.py
```

## Models used

- **Naive** – last value repeated  
- **Moving Average** – 7-day mean repeated  
- **ARIMA (2,1,2)** – statistical  
- **Exponential Smoothing (Holt)** – statistical  
- **LinearFeatures** – lag/rolling/calendar features (numpy least squares; stands in for RF/GB when scikit-learn is unavailable)

## Key operational files

- `outputs/forward_forecast_14day.csv` – next 14 days for both targets  
- `outputs/net_pressure_early_warning.csv` – capacity stress flags  
- `outputs/kpi_summary.csv` – best model per target  

Code is kept simple and readable for learners.
