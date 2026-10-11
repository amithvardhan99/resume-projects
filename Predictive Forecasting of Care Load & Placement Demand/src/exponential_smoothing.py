import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")
from misc_code import *


from statsmodels.tsa.holtwinters import SimpleExpSmoothing,Holt,ExponentialSmoothing

def select_exponential_smoothing_model(df,target_variable,size_of_window=7):
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

def train_exponential_smoothing_model(df,best_model_name,target_variable,size_of_window=7):
    model_configurations = pd.DataFrame([
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
	    ])

    model_config = model_configurations[model_configurations["name"]==best_model_name]
    series = df.set_index("Date")[target_variable].dropna()

    expo_smooth = ExponentialSmoothing(
        series,
        trend=model_config["trend"].values[0],
        damped_trend=model_config["damped_trend"].values[0],
        seasonal=model_config["seasonal"].values[0],
        seasonal_periods=(size_of_window if model_config["seasonal"].values[0] is not None else None),
        initialization_method="estimated"
    )

    model = expo_smooth.fit()

    return model


def multi_horizon_exponential_smoothing_error_profile(df,validation_window_1,training_dates,model_name,target_variable,size_of_window=7):

    validation_data = retrieve_validation_window(validation_window_1,size_of_window)

    max_horizon = max(
        (len(window) for window in validation_data),
        default=0
    )

    #print(validation_data)

    cols = [f"Day {i}" for i in range(1, max_horizon + 1)]

    mae_df = pd.DataFrame(columns=["Fold"] + cols)
    rmse_df = pd.DataFrame(columns=["Fold"] + cols)

    # Track previously observed validation dates
    validation_list = []

    for fold, validation_window in enumerate(validation_data):

        # Construct training dataset
        train_mask = (
            df["Date"].isin(training_dates)
            | df["Date"].isin(validation_list)
        )


        df_repeated_train = df.loc[train_mask].copy()

        # Fit the Exponential Smoothing model
        exp_smooth_fitted = train_exponential_smoothing_model(
            df_repeated_train,
            model_name,
            target_variable,
            size_of_window=size_of_window
        )

        # Forecast the validation window
        forecasted_values = exp_smooth_fitted.forecast(
            len(validation_window)
        )

        # Retrieve actual values in validation-window order
        actual_values = (
            df.loc[df["Date"].isin(validation_window)]
            .set_index("Date")
            .reindex(validation_window)[target_variable]
            .to_numpy()
        )

        predicted_values = np.asarray(forecasted_values)

        # Calculate errors
        absolute_errors = np.abs(actual_values - predicted_values)
        squared_errors = (actual_values - predicted_values) ** 2

        # Horizon-specific columns
        horizon_cols = [
            f"Day {i}"
            for i in range(1, len(validation_window) + 1)
        ]

        # Store fold number
        mae_df.loc[fold, "Fold"] = fold + 1
        rmse_df.loc[fold, "Fold"] = fold + 1

        # Store horizon-specific errors
        mae_df.loc[fold, horizon_cols] = np.round(
            absolute_errors, 2
        )

        rmse_df.loc[fold, horizon_cols] = np.round(
            squared_errors, 2
        )

        # Calculate fold-level MAE and RMSE
        mae_df.loc[fold, "MAE"] = np.round(
            np.mean(absolute_errors), 2
        )

        rmse_df.loc[fold, "RMSE"] = np.round(
            np.sqrt(np.mean(squared_errors)), 2
        )

        # Add observed validation dates for subsequent folds
        validation_list.extend(validation_window)

    # Average MAE for each forecast horizon
    mae_horizon_error_profile = (
        mae_df[cols].mean(axis=0).round(2)
    )

    # RMSE for each forecast horizon across folds
    rmse_horizon_error_profile = (
        np.power(rmse_df[cols].mean(axis=0),0.5).round(2)
    )

    return (
        mae_df,
        rmse_df,
        mae_horizon_error_profile,
        rmse_horizon_error_profile
    )