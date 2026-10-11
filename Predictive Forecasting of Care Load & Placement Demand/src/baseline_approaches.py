import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
import warnings
warnings.filterwarnings("ignore")
from misc_code import *


def naive_model(df,start_date,target_variable):
    predicted_value = df[df["Date"] == start_date][f"{target_variable}_lag_1"].values[0]
    return predicted_value

def calculate_moving_avg(df, window_dates, window_size, target_variable):
    return df[df["Date"].isin(window_dates)][f"{target_variable}_rolling_avg_{window_size}"].round(2).values


def multi_horizon_evaluation(df_7,validation_window_1,size_of_window,target_variable,method):
    validation_data = retrieve_validation_window(validation_window_1,size_of_window)
    cols = ["Day " + str(i) for i in range(1, size_of_window + 1)]

    mae_df = pd.DataFrame(columns=["Fold"] + cols)
    rmse_df = pd.DataFrame(columns=["Fold"] + cols)

    rt = 0

    for i in validation_data:
        start_date = i[0]
        df_train_test = pd.DataFrame(columns=["Date", f"Actual_{target_variable}",f"Predicted_{target_variable}"])
        # Generate predictions
        if method == "naive":
            target_variable_predicted_value = naive_model(df_7,start_date,target_variable)
        elif method == "moving average":
            target_variable_predicted_value = calculate_moving_avg(df_7,i,size_of_window,target_variable)
        else:
            raise ValueError("Invalid method. Use 'naive' or 'moving average'.")

        # Actual and predicted values
        df_train_test["Date"] = i
        df_train_test[f"Actual_{target_variable}"] = df_7[df_7["Date"].isin(i)][target_variable].values

        df_train_test[f"Predicted_{target_variable}"] = target_variable_predicted_value

        cols_1 = ["Day "+str(j) for j in range(1,len(i)+1)]

        # Calculate absolute errors
        df_train_test["Absolute_Error"] = abs(df_train_test[f"Predicted_{target_variable}"]-df_train_test[f"Actual_{target_variable}"])

        mae = mean_absolute_error(df_train_test[f"Actual_{target_variable}"],df_train_test[f"Predicted_{target_variable}"])

        mae_df.loc[rt, "Fold"] = rt + 1
        mae_df.loc[rt, cols_1] = df_train_test["Absolute_Error"].values
        mae_df.loc[rt, "MAE"] = round(mae,2)

        # Calculate squared errors
        df_train_test["Squared_Error"] = (df_train_test[f"Predicted_{target_variable}"] - df_train_test[f"Actual_{target_variable}"]) ** 2

        rmse = root_mean_squared_error(df_train_test[f"Actual_{target_variable}"],df_train_test[f"Predicted_{target_variable}"])

        rmse_df.loc[rt, "Fold"] = rt + 1
        rmse_df.loc[rt, cols_1] = df_train_test["Squared_Error"].values
        rmse_df.loc[rt, "RMSE"] = round(rmse,2)

        rt += 1

    # MAE by forecast horizon
    mae_horizon_error_profile = mae_df.mean()
    mae_horizon_error_profile.drop(["Fold", "MAE"],axis=0,inplace=True)
    mae_horizon_error_profile = mae_horizon_error_profile.round(2)

    # RMSE by forecast horizon
    rmse_horizon_error_profile = rmse_df.mean()
    rmse_horizon_error_profile.drop(["Fold", "RMSE"],axis=0,inplace=True)
    rmse_horizon_error_profile = np.sqrt(rmse_horizon_error_profile).round(2)

    return (mae_df,rmse_df,mae_horizon_error_profile,rmse_horizon_error_profile)