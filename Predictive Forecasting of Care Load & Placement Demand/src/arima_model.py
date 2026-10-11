import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
import itertools
from statsmodels.tsa.stattools import acf, pacf
from misc_code import *

warnings.filterwarnings("ignore")

def differencing_method(df,target_variable):
    df_i = df.copy()
    i = 0
    while True:
        result = adfuller(df_i[f"{target_variable}_diff_"+str(i)])
        adf = result[0]
        p_value = result[1]
        print("Stationarity measure (p-value): ",round(p_value,2))
        if p_value <= 0.05:
            print("Series is stationary, no need to perform differencing")
            break
        else:
            print("Series is not stationary, we need to perform differencing")
            df_i[f"{target_variable}_diff_"+str(i+1)] = df_i[f"{target_variable}_diff_"+str(i)].diff()
            df_i = df_i.dropna()
            i += 1
            continue
    d = i
    return df_i,f"{target_variable}_diff_"+str(i),d

def perform_differencing(df_7,target_variable):
    df_i = df_7[["Date", target_variable]]
    df_i[f"{target_variable}_diff_0"] = df_i[target_variable].copy()
    df_i["Date"] = pd.to_datetime(df_i["Date"])
    df_i, name_of_stationary_column,d = differencing_method(df_i,target_variable)
    return df_i, name_of_stationary_column,d


def find_p_and_q_values(df_i, name_of_stationary_column, max_lags=30, alpha=0.05):
    series = df_i[name_of_stationary_column].dropna()
    max_lags = min(max_lags, len(series)//2-1)

    if max_lags < 1:
        raise ValueError("Insufficient observations to calculate PACF and ACF.")

    acf_values = acf(series, nlags=max_lags, alpha=alpha)
    pacf_values = pacf(series, nlags=max_lags, alpha=alpha, method="ywm")

    acf_coefficients = acf_values[0]
    acf_confidence = acf_values[1]

    pacf_coefficients = pacf_values[0]
    pacf_confidence = pacf_values[1]

    acf_significant_lags = [lag for lag in range(1,max_lags+1) if not (acf_confidence[lag, 0]<=0<=acf_confidence[lag, 1])]

    pacf_significant_lags = [lag for lag in range(1,max_lags+1) if not (pacf_confidence[lag, 0]<=0<=pacf_confidence[lag, 1])]

    # Display significant lags
    # print("Significant ACF lags (candidate q values):",
         #  acf_significant_lags)

    # print("Significant PACF lags (candidate p values):",
          # pacf_significant_lags)

    p = pacf_significant_lags[0:3]
    q = acf_significant_lags[0:3]

    return p,q

def retrieve_arima_model(df,p,d,q,target_variable):
    try:
        model = ARIMA(df[target_variable],order=(p,d,q))
        return model
    except:
        return 0

def calculate_arima_model(df,size_of_window,p,d,q,target_variable):
    arima_model = retrieve_arima_model(df,p,d,q,target_variable)
    final_model = arima_model.fit()
    predictions = final_model.forecast(steps=size_of_window)
    return predictions

def multi_horizon_arima_evaluation(df_7,target_variable,p,d,q,training_dates,validation_window_1,size_of_window=7):
    validation_data = retrieve_validation_window(validation_window_1, size_of_window)
    no_of_validation_windows = len(validation_data)
    cols = ["Window " + str(i + 1) for i in range(no_of_validation_windows)]
    cols_mae = ["Model"] + cols + ["Average MAE"]
    cols_rmse = ["Model"] + cols + ["Average RMSE"]

    df_arima_mae = pd.DataFrame(columns=cols_mae)
    df_arima_rmse = pd.DataFrame(columns=cols_rmse)

    list_of_models = []
    no_of_validation_dates_for_models = []

    ctr = 0
    combination_of_p_and_q_values = list(itertools.product(p, q))

    df_arima_mae["Model"] = combination_of_p_and_q_values.copy()
    df_arima_rmse["Model"] = combination_of_p_and_q_values.copy()

    for i in combination_of_p_and_q_values:
        for num in range(0, len(validation_data)):
            no_of_steps = len(validation_data[num])
            # print(no_of_steps)
            df_w = df_7[
                [k in training_dates or k in validation_window_1[0:size_of_window * num] for k in df_7["Date"]]].copy()
            # model,AIC,order = retrieve_arima_model(df_w)
            model = retrieve_arima_model(df_w, i[0], d, i[1], target_variable)
            result = model.fit()
            predicted_values = result.forecast(steps=no_of_steps)
            list_of_models.append(model)
            no_of_validation_dates_for_models.append(size_of_window * num)

            mae = mean_absolute_error(df_7.loc[predicted_values.index.values, target_variable], predicted_values)
            rmse = root_mean_squared_error(df_7.loc[predicted_values.index.values, target_variable], predicted_values)

            df_arima_mae.loc[ctr, "Window " + str(num + 1)] = mae
            df_arima_rmse.loc[ctr, "Window " + str(num + 1)] = rmse
        ctr += 1

    for i in df_arima_mae.index:
        df_arima_mae.loc[i, "Average MAE"] = df_arima_mae.loc[i, cols].mean()
        df_arima_rmse.loc[i, "Average RMSE"] = df_arima_rmse.loc[i, cols].mean()

    position_of_best_model = int(np.where(df_arima_mae["Average MAE"] == df_arima_mae["Average MAE"].min())[0][0])
    p1, q1 = df_arima_mae.loc[position_of_best_model, "Model"]
    return p1,q1

def multi_horizon_arima_error_profile(df,target_variable,training_dates,validation_window_1,p,d,q,size_of_window=7):
    validation_data = retrieve_validation_window(validation_window_1,size_of_window)

    cols = ["Day " + str(i) for i in range(1, size_of_window + 1)]

    mae_df = pd.DataFrame(columns=["Fold"] + cols)
    rmse_df = pd.DataFrame(columns=["Fold"] + cols)

    validation_dates = []

    for rt, validation_window in enumerate(validation_data):
        no_of_steps = len(validation_window)
        cols_1 = ["Day " + str(i) for i in range(1, no_of_steps + 1)]

        df_w = df.loc[df["Date"].isin(list(training_dates) + validation_dates)].copy()

        df_w = df_w.sort_values("Date")

        predictions = calculate_arima_model(df_w,no_of_steps,p,d,q,target_variable)

        df_c = pd.DataFrame()
        df_c["Actual"] = df.loc[df["Date"].isin(validation_window), target_variable].values
        df_c["Predicted"] = np.asarray(predictions)
        df_c["Absolute Error"] = abs(df_c["Actual"] - df_c["Predicted"])
        df_c["Squared Error"] = (df_c["Actual"] - df_c["Predicted"]) ** 2
        mae = df_c["Absolute Error"].mean()
        rmse = np.sqrt(df_c["Squared Error"].mean())
        mae_df.loc[rt, "Fold"] = rt + 1
        mae_df.loc[rt, cols_1] = (df_c["Absolute Error"].values)
        mae_df.loc[rt, "MAE"] = round(mae, 2)
        rmse_df.loc[rt, "Fold"] = rt + 1
        rmse_df.loc[rt, cols_1] = (df_c["Squared Error"].values)
        rmse_df.loc[rt, "RMSE"] = round(rmse, 2)
        validation_dates.extend(validation_window)

    mae_horizon_error_profile = (mae_df[cols].mean()).round(2)

    # Calculate RMSE for each forecast horizon
    rmse_horizon_error_profile = np.power(rmse_df[cols].mean(),0.5).round(2)

    return (
        mae_df,
        rmse_df,
        mae_horizon_error_profile,
        rmse_horizon_error_profile
    )
