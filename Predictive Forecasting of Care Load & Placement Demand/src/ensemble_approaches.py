from misc_code import *
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor

def consider_only_lag_columns(df_7):
    all_columns = pd.Series(df_7.columns.to_numpy())

    placed_in_cbp_lags = all_columns[all_columns.str.startswith("Placed_In_CBP")].values.tolist()
    cbp_custody_lags = all_columns[all_columns.str.startswith("CBP_Custody")].values.tolist()
    cbp_to_hhs_lags = all_columns[all_columns.str.startswith("CBP_To_HHS")].values.tolist()
    hhs_care_lags = all_columns[all_columns.str.startswith("HHS_Care")].values.tolist()
    discharged_from_hhs_lags = all_columns[all_columns.str.startswith("Discharged_From_HHS")].values.tolist()

    lag_columns = ["Date"] + placed_in_cbp_lags + cbp_custody_lags + cbp_to_hhs_lags + hhs_care_lags + discharged_from_hhs_lags + [
        "Day_Of_Week", "Month", "Is_Weekend"]

    df_lag = df_7[lag_columns]

    return df_lag

def retrieve_random_forest_model(df,target_variable):
    X = df.drop(columns=[target_variable])
    y = df[target_variable]
    random_forest_regr = RandomForestRegressor()
    random_forest_model = random_forest_regr.fit(X,y)
    return random_forest_model

def retrieve_gradient_boosting_model(df,target_variable):
    X = df.drop(columns=[target_variable])
    y = df[target_variable]
    gradient_boosting_regr = GradientBoostingRegressor()
    gradient_boosting_model = gradient_boosting_regr.fit(X,y)
    return gradient_boosting_model

def multi_horizon_ensemble_error_profile(df_lag,validation_window_1,target_variable,ml_algo,size_of_window=7):

    # Initialize evaluation DataFrames
    cols = ["Day " + str(i) for i in range(1, size_of_window + 1)]

    df_lag_train = df_lag[[k in training_dates for k in df_lag["Date"]]].copy()

    df_lag_valid = df_lag[[k in validation_window_1 for k in df_lag["Date"]]].copy()

    validation_data = retrieve_validation_window(validation_window_1,size_of_window)

    mae_df = pd.DataFrame(columns=["Fold"] + cols)
    rmse_df = pd.DataFrame(columns=["Fold"] + cols)

    # Store validation dates already processed
    processed_valid_dates = []

    # Store actual and predicted values
    uf = pd.DataFrame(
        columns=["Date", "Actual", "Predicted"]
    )

    t = 0

    for i in validation_data:

        # Retrieve previously processed validation dates
        df_lag_valid_dates = df_lag_valid[
            df_lag_valid["Date"].isin(processed_valid_dates)
        ].copy()

        # Combine training data with previously processed
        # validation data
        df_input = pd.concat(
            [df_lag_train, df_lag_valid_dates],
            axis=0
        ).drop(columns="Date")

        # Train the Random Forest model
        if ml_algo == "random forest":
            model = retrieve_random_forest_model(
                df_input,
                target_variable
            )
        elif ml_algo == "gradient boosting":
            model = retrieve_gradient_boosting_model(
                df_input,
                target_variable
            )
        else:
            raise ValueError("Invalid method. Use 'random forest' or 'gradient boosting'.")

        # Retrieve the current validation window
        test = df_lag_valid[
            df_lag_valid["Date"].isin(i)
        ].copy()

        # Separate features and target
        X_test = test.drop(
            columns=["Date", target_variable]
        )

        y_test = test[target_variable]

        # Generate predictions
        y_pred = model.predict(X_test)

        # Create prediction DataFrame
        po = pd.DataFrame({
            "Date": test["Date"].values,
            "Actual": y_test.values,
            "Predicted": y_pred
        })

        # Calculate errors
        po["Absolute Error"] = np.abs(
            po["Actual"] - po["Predicted"]
        )

        po["Squared Error"] = (
            po["Actual"] - po["Predicted"]
        ) ** 2

        # Generate horizon column names for the current window
        l = len(y_pred)
        cols = ["Day " + str(j) for j in range(1, l + 1)]

        # Store fold number
        mae_df.loc[t, "Fold"] = t + 1
        rmse_df.loc[t, "Fold"] = t + 1

        # Store daily absolute and squared errors
        mae_df.loc[t, cols] = po["Absolute Error"].values
        rmse_df.loc[t, cols] = po["Squared Error"].values

        # Calculate fold-level MAE and RMSE
        mae_df.loc[t, "MAE"] = (
            po["Absolute Error"].mean().round(2)
        )

        rmse_df.loc[t, "RMSE"] = np.round(
            np.sqrt(po["Squared Error"].mean()), 2
        )

        # Append predictions and errors
        uf = pd.concat(
            [uf, po],
            axis=0,
            ignore_index=True
        )

        # Mark current validation dates as processed
        processed_valid_dates.extend(i)

        t += 1

    # Calculate horizon-wise MAE profile
    mae_horizon_error_profile = (
        mae_df.drop(columns=["Fold", "MAE"], errors="ignore")
        .mean()
        .round(2)
    )

    # Calculate horizon-wise RMSE profile
    rmse_horizon_error_profile = np.power(
        rmse_df.drop(
            columns=["Fold", "RMSE"],
            errors="ignore"
        ).mean(), 0.5
    ).round(2)

    # Return evaluation results
    return (
        mae_df,
        rmse_df,
        mae_horizon_error_profile,
        rmse_horizon_error_profile
    )



