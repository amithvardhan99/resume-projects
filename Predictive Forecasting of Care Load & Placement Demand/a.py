import numpy as np
import pandas as pd


def evaluate_exponential_smoothing(df, validation_windows, training_dates, model_name, target_variable,
                                   size_of_window=7):
    max_horizon = max(
        (len(window) for window in validation_windows),
        default=0
    )

    cols = [f"Day {i}" for i in range(1, max_horizon + 1)]

    mae_df = pd.DataFrame(columns=["Fold"] + cols)
    rmse_df = pd.DataFrame(columns=["Fold"] + cols)

    # Track previously observed validation dates
    validation_list = []

    for fold, validation_window in enumerate(validation_windows):
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
            target_col,
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
        np.sqrt(rmse_df[cols].mean(axis=0)).round(2)
    )

    return (
        mae_df,
        rmse_df,
        mae_horizon_error_profile,
        rmse_horizon_error_profile
    )


(
    mae_df,
    rmse_df,
    mae_horizon_error_profile,
    rmse_horizon_error_profile
) = evaluate_exponential_smoothing(
    df=df_7,
    validation_windows=validation_data,
    training_dates=training_dates,
    model_name=best_model_name,
    target_col="HHS_Care",
    size_of_window=7
)
```

** Note: ** The
date
column is now
hardcoded as `"Date"`, and you
no
longer
need
to
pass
`date_col`
when
calling
the
function.
