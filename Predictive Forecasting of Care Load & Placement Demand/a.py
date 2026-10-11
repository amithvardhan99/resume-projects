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

    p = pacf_significant_lags[0:4]
    q = acf_significant_lags[0:4]

    # Display significant lags
    print("Significant ACF lags (candidate q values):",
          p)

    print("Significant PACF lags (candidate p values):",
          q)

    return p,q