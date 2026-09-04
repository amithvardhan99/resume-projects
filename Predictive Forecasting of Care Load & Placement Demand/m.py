    size_of_window = 7
def retrieve_mse_and_rmse_
    validation_data = retrieve_validation_window(validation_window_1,size_of_window)
    cols = ["Day "+str(i) for i in range(1,size_of_window+1)]
    mae_df = pd.DataFrame(columns = ["Fold"]+cols)
    rmse_df = pd.DataFrame(columns = ["Fold"]+cols)
    rt = 0

    for i in validation_data:
        start_date = i[0]
        # print(i)
        df_train_test = pd.DataFrame(columns=["Date","Actual_HHS_Care","Predicted_HHS_Care"])

        hhs_care_predicted_value = naive_model(df_7,start_date)

        df_train_test["Date"] = i
        df_train_test["Actual_HHS_Care"] = df_7[[k in i for k in df_7["Date"]]]["HHS_Care"].values
        df_train_test["Predicted_HHS_Care"] = hhs_care_predicted_value

        cols_1 = ["Day "+str(i) for i in range(1,len(i)+1)]


        df_train_test["Absolute_Error"] = abs(df_train_test["Predicted_HHS_Care"] - df_train_test["Actual_HHS_Care"])
        mae = mean_absolute_error(df_train_test["Actual_HHS_Care"],df_train_test["Predicted_HHS_Care"])
        mae_df.loc[rt,"Fold"] = (rt + 1)
        mae_df.loc[rt,cols_1] = df_train_test["Absolute_Error"].values
        mae_df.loc[rt,"MAE"] = round(mae,2)


        df_train_test["Squared_Error"] = (df_train_test["Predicted_HHS_Care"] - df_train_test["Actual_HHS_Care"]) * (df_train_test["Predicted_HHS_Care"] - df_train_test["Actual_HHS_Care"])
        rmse = root_mean_squared_error(df_train_test["Actual_HHS_Care"],df_train_test["Predicted_HHS_Care"])
        rmse_df.loc[rt,"Fold"] = (rt + 1)
        rmse_df.loc[rt,cols_1] = df_train_test["Squared_Error"].values
        rmse_df.loc[rt,"RMSE"] = round(rmse,2)


        rt += 1

    mae_horizon_error_profile = mae_df.mean()
    mae_horizon_error_profile.drop(["Fold","MAE"],axis=0,inplace=True)
    mae_horizon_error_profile = mae_horizon_error_profile.round(2)

    rmse_horizon_error_profile = rmse_df.mean()
    rmse_horizon_error_profile.drop(["Fold","RMSE"],axis=0,inplace=True)
    rmse_horizon_error_profile = round(np.power(rmse_horizon_error_profile,0.5),2)