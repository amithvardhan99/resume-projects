import pandas as pd

from statsmodels.tsa.holtwinters import SimpleExpSmoothing, Holt, ExponentialSmoothing


def select_exponential_smoothing_model(df, target_variable, size_of_window=7):
    series = df.set_index("Date")[target_variable].dropna()
    fitted_models = {}
    model_results = []

    model_configurations = [
        {
            "name": "Simple Exponential Smoothing",
            "trend": None,
            "damped_trend": False,
            "seasonal": None
        },
        {
            "name": "Holt Linear Trend",
            "trend": "add",
            "damped_trend": False,
            "seasonal": None
        },
        {
            "name": "Holt Damped Trend",
            "trend": "add",
            "damped_trend": True,
            "seasonal": None
        },
        {
            "name": "Holt-Winters Additive",
            "trend": "add",
            "damped_trend": False,
            "seasonal": "add"
        },
        {
            "name": "Holt-Winters Multiplicative",
            "trend": "add",
            "damped_trend": False,
            "seasonal": "mul"
        }
    ]

    for config in model_configurations:
        name = config["name"]
        try:
            if config["seasonal"] is not None:
                if len(series) < 2 * size_of_window:
                    print(f"Skipping {name}: insufficient observations for seasonal initialization.")
                    continue

            if config["seasonal"] == "mul" and (series <= 0).any():
                print(f"Skipping {name}: multiplicative seasonality requires positive observations.")
                continue

            model = ExponentialSmoothing(
                series,
                trend=config["trend"],
                damped_trend=config["damped_trend"],
                seasonal=config["seasonal"],
                seasonal_periods=(
                    size_of_window
                    if config["seasonal"] is not None
                    else None
                ),
                initialization_method="estimated"
            )

            fitted_model = model.fit(optimized=True)

            fitted_models[name] = fitted_model

            model_results.append({
                "Model": name,
                "AIC": fitted_model.aic,
                "BIC": fitted_model.bic,
                "SSE": fitted_model.sse
            })

        except (ValueError, ArithmeticError, np.linalg.LinAlgError) as e:
            print(f"Could not fit {name}: {e}")

    results_df = pd.DataFrame(model_results)

    if results_df.empty:
        raise ValueError("None of the exponential smoothing models could be fitted.")

    results_df = results_df.sort_values("AIC").reset_index(drop=True)
    best_model_name = results_df.loc[0, "Model"]

    return best_model_name