# Predictive Forecasting of Care Load & Placement Demand

## What this project does

This project predicts three things for the UAC (Unaccompanied Alien Children) program:

1. **How many children will be in HHS care** in the coming days  
2. **How many children will leave HHS care** (successful placements with sponsors)  
3. **Whether more children are coming in than leaving** (this is called net pressure and acts as an early warning)

The goal is to help planners act early instead of waiting until shelters or staff are already under stress.

---

## What is inside this folder

| Item | What it is |
|------|------------|
| `complete_forecasting.py` | Main Python script that builds all forecasts |
| `app/streamlit_app.py` | Interactive web dashboard |
| `outputs/` | All result files (tables and a simple web page) |
| `final_dataset.csv` | Clean daily data ready for forecasting |
| `notebook.ipynb` | Step-by-step analysis and model work |
| `src/` | Helper code for different forecast models |
| `Research_Paper_Care_Load_Forecasting.docx` | Full written report |
| `Executive_Summary_Government_Stakeholders.docx` | Short summary for leaders |
| `requirements.txt` | List of Python packages needed |
| `Book1.twb`, `Book2.twb`, `Book3.twb` | Tableau charts used for exploration |

---

## How to run the project

### Step 1 — Install the needed packages

```bash
pip install -r requirements.txt
```

### Step 2 — Create the forecasts

```bash
python complete_forecasting.py
```

This script reads the data, trains the models, and saves results in the `outputs` folder.

### Step 3 — Look at the results

**Option A (easiest):**  
Open `outputs/dashboard.html` in any web browser. No extra tools needed.

**Option B (interactive):**  
```bash
streamlit run app/streamlit_app.py
```
Then open the link that appears in the terminal (usually http://localhost:8501).

---

## Models used in this project

- **Naive** — uses the last known number and repeats it  
- **Moving Average** — uses the average of the last 7 days  
- **ARIMA** — a common statistical time-series model  
- **Exponential Smoothing** — another statistical model that follows trends  
- **LinearFeatures** — uses past values and rolling averages as inputs (simple machine-learning style)

---

## Important result files

| File | What it shows |
|------|----------------|
| `outputs/forward_forecast_14day.csv` | Predicted numbers for the next 14 days |
| `outputs/net_pressure_early_warning.csv` | Whether intake is higher than discharges |
| `outputs/model_metrics.csv` | How accurate each model was |
| `outputs/kpi_summary.csv` | Which model worked best for each target |
| `outputs/test_predictions.csv` | Actual vs predicted values used for checking accuracy |

---

## Short note on decimal numbers

Forecasts can show numbers like 11.1 or 12.7. Real children are whole numbers. The models work with continuous math, so decimals appear. For planning, round them to the nearest whole number.

---

## Summary

This project turns daily UAC data into short-term forecasts and a simple early-warning signal. It includes data cleaning, feature building, several forecast models, a dashboard, a research paper, and an executive summary — all in one place.
