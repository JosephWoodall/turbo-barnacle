import random
from datetime import datetime, timedelta
import statsmodels.api as sm
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from fbprophet import Prophet


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
    elif model_type == 'Exponential Smoothing':
        model = sm.tsa.holtwinters.ExponentialSmoothing(
            revenues, seasonal='add', trend='mul')
    elif model_type == 'Prophet':
        df = pd.DataFrame({'ds': dates, 'y': revenues})
        model = Prophet()
        model.fit(df)
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
    if model_type in ['ARIMA', 'SARIMA', 'Exponential Smoothing']:
        forecast = model_fit.forecast(steps=5)
    elif model_type == 'Prophet':
        future = model.make_future_dataframe(periods=5)
        forecast = model.predict(future)["yhat"]
    elif model_type in ['Linear Regression', 'Random Forest']:
        future_dates = [dates[-1] + timedelta(days=i) for i in range(1, 6)]
        forecast = model.predict(future_dates.reshape(-1, 1))
    else:
        raise ValueError("Invalid model type for forecasting")

    return forecast


data = generate_financial_data(datetime(2020, 1, 1), datetime(2020, 12, 31))
print(data[:5])

models = ['ARIMA', 'SARIMA', 'Exponential Smoothing',
          'Prophet', 'Linear Regression', 'Random Forest']
for model in models:
    forecast = perform_time_series_forecasting(data, model)
    print(f"Forecast for {model}: {forecast}")
