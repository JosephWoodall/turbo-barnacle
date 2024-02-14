import random
from datetime import datetime, timedelta
import statsmodels.api as sm


def generate_financial_data(start_date, end_date):
    data = []
    current_date = start_date

    while current_date <= end_date:
        revenue = random.randint(1000, 10000)
        data.append((current_date, revenue))
        current_date += timedelta(days=1)

    return data


def perform_time_series_forecasting(data, model_type):
    dates = [d[0] for d in data]
    revenues = [d[1] for d in data]

    # Create a time series model
    if model_type == 'ARIMA':
        model = sm.tsa.ARIMA(revenues, order=(1, 0, 0))
    elif model_type == 'SARIMA':
        model = sm.tsa.SARIMAX(revenues, order=(1, 0, 0), seasonal_order=(1, 0, 0, 12))
    else:
        raise ValueError("Invalid model type")

    # Fit the model to the data
    model_fit = model.fit()

    # Perform forecasting
    forecast = model_fit.forecast(steps=5)

    return forecast


data = generate_financial_data(datetime(2020, 1, 1), datetime(2020, 12, 31))
print(data[:5])

forecast = perform_time_series_forecasting(data, 'SARIMA')
print(forecast)

