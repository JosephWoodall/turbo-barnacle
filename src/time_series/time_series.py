import random
from datetime import datetime, timedelta
import statsmodels.api as sm
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt


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
        model = sm.tsa.SARIMAX(revenues, order=(
            1, 0, 0), seasonal_order=(1, 0, 0, 12))
    elif model_type == 'Linear Regression':
        model = LinearRegression()
        model.fit(dates.reshape(-1, 1), revenues)
    elif model_type == 'Random Forest':
        model = RandomForestRegressor()
        model.fit(dates.reshape(-1, 1), revenues)
    else:
        raise ValueError("Invalid model type")

    # Fit the model to the data
    model_fit = model.fit()

    # Perform forecasting
    if model_type in ['ARIMA', 'SARIMA']:
        forecast = model_fit.forecast(steps=5)
    elif model_type in ['Linear Regression', 'Random Forest']:
        future_dates = [dates[-1] + timedelta(days=i) for i in range(1, 6)]
        forecast = model.predict(future_dates.reshape(-1, 1))
    else:
        raise ValueError("Invalid model type for forecasting")

    return forecast


def plot_forecast(forecast):
    plt.plot(forecast)
    plt.xlabel('Time')
    plt.ylabel('Forecasted Values')
    plt.title('Forecasted Values Over Time')
    plt.show()


data = generate_financial_data(datetime(2020, 1, 1), datetime(2020, 12, 31))
print(data[:5])

models = ['ARIMA', 'SARIMA', 'Exponential Smoothing',
          'Linear Regression', 'Random Forest']
for model in models:
    forecast = perform_time_series_forecasting(data, model)
    print(f"Forecast for {model}: {forecast}")
    plot_forecast(forecast)
