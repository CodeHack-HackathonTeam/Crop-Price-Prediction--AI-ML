import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error

# Load and prepare the data
@st.cache
def load_data():
    # Assuming the combined dataset is in 'combined_dataset.csv'
    data = pd.read_csv('combined_dataset.csv', parse_dates=['DATE'])
    data.set_index('DATE', inplace=True)  # Ensure 'DATE' is the index
    return data

data = load_data()

# Streamlit Interface for user input
st.title("Commodity Price Forecasting using SARIMA")

# Select the commodity and state for forecasting
commodity = st.selectbox("Select Commodity:", data['Commodity'].unique())
state = st.selectbox("Select State/UT:", data['States/UTs'].unique())

# Filter the data for selected commodity and state
commodity_data = data[(data['Commodity'] == commodity) & (data['States/UTs'] == state)]

# Plot the historical data for the selected commodity and state
st.subheader(f"Historical Price Data for {commodity} in {state}")
st.line_chart(commodity_data['Price'])

# Split the data into training and testing sets (80% train, 20% test)
train_size = int(len(commodity_data) * 0.8)
train, test = commodity_data[:train_size], commodity_data[train_size:]

# Ensure the date index has frequency information
train.index = pd.to_datetime(train.index)
test.index = pd.to_datetime(test.index)

# Set frequency to 'D' (daily), adjust if necessary (e.g., weekly 'W', monthly 'M')
train = train.asfreq('D')
test = test.asfreq('D')

# Fit the SARIMA model on the training data
sarima_model = SARIMAX(train['Price'],
                       order=(1, 1, 1),  # AR(1), I(1), MA(1)
                       seasonal_order=(1, 1, 1, 7),  # Seasonal AR, I, MA with weekly seasonality
                       enforce_stationarity=False,
                       enforce_invertibility=False)

# Fit the model
sarima_result = sarima_model.fit()

# Get forecasted values
forecast_steps = len(test)
forecast = sarima_result.get_forecast(steps=forecast_steps)
forecast_index = test.index

# Get the predicted values and confidence intervals
forecast_values = forecast.predicted_mean
forecast_conf_int = forecast.conf_int()

# Plot the forecasted values along with the historical data
st.subheader(f"Forecasted Prices for {commodity} in {state}")
plt.figure(figsize=(10, 6))
plt.plot(train.index, train['Price'], label='Training Data', color='blue')
plt.plot(test.index, test['Price'], label='Test Data', color='orange')
plt.plot(forecast_index, forecast_values, label='Forecasted Price', color='red')
plt.fill_between(forecast_index, forecast_conf_int.iloc[:, 0], forecast_conf_int.iloc[:, 1], color='pink', alpha=0.3)
plt.title(f'{commodity} Price Forecast for {state}')
plt.xlabel('Date')
plt.ylabel('Price (INR)')
plt.legend()
st.pyplot()

# Evaluate the SARIMA Model
test_predictions = sarima_result.predict(start=test.index[0], end=test.index[-1])

# Handling NaN values before calculating RMSE
test_predictions = test_predictions.dropna()
test_actual = test['Price'].dropna()

# Ensure no length mismatch between predictions and actual values
if len(test_actual) == len(test_predictions):
    mse = mean_squared_error(test_actual, test_predictions)
    rmse = np.sqrt(mse)
else:
    st.error("Mismatch in the number of data points between the test data and predictions.")
    rmse = None

# Display RMSE if no mismatch
if rmse is not None:
    st.subheader("Model Evaluation")
    st.write(f"Root Mean Squared Error (RMSE) on Test Set: {rmse:.2f}")

# Display forecasted values
st.subheader(f"Forecasted Values for the next {forecast_steps} Days")
forecast_df = pd.DataFrame({'Date': forecast_index, 'Predicted Price (INR)': forecast_values})
st.write(forecast_df)
