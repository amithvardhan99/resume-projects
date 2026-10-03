from datetime import datetime, timedelta
from statsmodels.tsa.seasonal import seasonal_decompose
import numpy as np
import pandas as pd

from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.stats.diagnostic import acorr_ljungbox
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
import itertools

import warnings


from tensorflow.core.distributed_runtime import preemption
from statsmodels.tsa.api import ExponentialSmoothing,SimpleExpSmoothing,Holt

warnings.filterwarnings("ignore")


df_7 = pd.read_csv("final_dataset.csv")


initial_day = df_7.loc[0,"Date"]
final_day = df_7.loc[df_7.shape[0]-1,"Date"]

td = (final_day - initial_day)
total_number_of_days = td.days

number_of_training_dates = round(total_number_of_days * 67 / 100)
size_of_validation_window = round(total_number_of_days * 25 / 100)
number_of_testing_dates = total_number_of_days - (number_of_training_dates + size_of_validation_window)

training_threshold = initial_day + timedelta(number_of_training_dates - 1)
validation_threshold = training_threshold + timedelta(size_of_validation_window - 1)

training_dates = pd.date_range(
    initial_day,
    training_threshold,
    inclusive = "left",
)

validation_window_1 = pd.date_range(
    training_threshold,
    validation_threshold,
    inclusive = "left"
)

testing_dates = pd.date_range(
    validation_threshold,
    final_day,
    inclusive = "both"
)



def retrieve_validation_window(validation_window_1,length_of_window):
	validation_data = []
	v = 0
	p = []
    for i in validation_window_1:
        if v == length_of_window:
            t = pd.date_range(p[0],p[-1])
            validation_data.append(t)
            p = []
            v = 0
        p.append(i)
        v += 1

    if p is not None:
        t = pd.date_range(p[0],p[-1])
        validation_data.append(t)

	return validation_data