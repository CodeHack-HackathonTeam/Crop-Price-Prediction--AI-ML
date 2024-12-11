import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Load preprocessed data
data = pd.read_csv('preprocessed_data.csv', parse_dates=['DATE'])

# Feature engineering: create additional lag features for 'Potato'
data['Potato_Lag_7'] = data['Potato'].shift(7)
data['Potato_Lag_14'] = data['Potato'].shift(14)
data['Potato_Lag_30'] = data['Potato'].shift(30)

# Drop rows with missing values due to shifting
data.dropna(subset=['Potato_Lag_7', 'Potato_Lag_14', 'Potato_Lag_30'], inplace=True)

# Define features (X) and target variable (y)
X = data[['Potato_Lag_7', 'Potato_Lag_14', 'Potato_Lag_30']]  # Add more features if needed
y = data['Potato']  # Target: Price of Potato

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Hyperparameter tuning using GridSearchCV
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# Initialize the RandomForest model
rf = RandomForestRegressor(random_state=42)

# GridSearchCV to find best parameters
grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2)
grid_search.fit(X_train, y_train)

# Get the best model
best_rf = grid_search.best_estimator_

# Make predictions using the best model
y_pred = best_rf.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
rmse = mse ** 0.5
print(f"Root Mean Squared Error (RMSE): {rmse}")

# Plotting Actual vs Predicted Prices for Evaluation
plt.figure(figsize=(10, 5))
plt.plot(y_test.index, y_test, label='Actual Prices', color='blue')
plt.plot(y_test.index, y_pred, label='Predicted Prices', color='red', linestyle='--')
plt.title('Actual vs Predicted Prices')
plt.xlabel('Date')
plt.ylabel('Price (INR)')
plt.legend()
plt.show()

# Streamlit Interface
st.title("Agri-Horticultural Price Prediction Dashboard")

# Sidebar for user input
st.sidebar.header('User Input Parameters')
input_lag = st.sidebar.slider('Select Lag (Days)', min_value=1, max_value=30, value=7)
input_days_ahead = st.sidebar.slider('Predict for how many days ahead?', min_value=1, max_value=30, value=7)

# Function to predict for future dates for each state
def predict_price_for_state(state, lag_days, days_ahead):
    state_data = data[data['States/UTs'] == state]  # Filter data for the selected state
    
    # Get the most recent price and feature
    recent_data = state_data.iloc[-1:]  # Last available data point
    recent_price = recent_data['Potato'].values[0]
    
    # Create future dates
    future_dates = pd.date_range(start=state_data['DATE'].iloc[-1], periods=days_ahead + 1, freq='D')[1:]

    # Predict future prices based on the selected lag
    future_predictions = []
    for _ in range(days_ahead):
        # Prepare the features (using the lag features of the most recent price)
        features = [[recent_price, recent_price, recent_price]]  # Using all lags for prediction
        future_price = best_rf.predict(features)[0]
        future_predictions.append(future_price)
        
        # Update the most recent price for the next prediction
        recent_price = future_price

    return future_dates, future_predictions

# Get the list of unique states
states = data['States/UTs'].unique()

# Initialize a dictionary to store predictions for each state
all_predictions = {}

# Predict for each state
for state in states:
    predicted_dates, predicted_prices = predict_price_for_state(state, input_lag, input_days_ahead)
    all_predictions[state] = pd.DataFrame({
        'Date': predicted_dates,
        'Predicted Price (INR)': predicted_prices
    })

# Display the state-wise predictions
st.subheader(f"State-wise Predicted Prices for the Next {input_days_ahead} Days (Lag: {input_lag} Days Ago)")

# Allow users to select the state they want to view predictions for
selected_state = st.sidebar.selectbox("Select State/UT for Prediction", states)

# Show predictions for the selected state
st.write(f"Predicted Prices for {selected_state}:")
st.write(all_predictions[selected_state])

# Display RMSE on Streamlit
st.subheader("Model Evaluation")
st.write(f"Root Mean Squared Error (RMSE): {rmse:.2f}")

# Correlation Matrix (for all commodities)
st.subheader("Correlation Matrix of All Commodities")
commodities = data.columns[2:]  # List of all commodity columns
corr_matrix = data[commodities].corr()

fig, ax = plt.subplots(figsize=(12, 10))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", ax=ax)
st.pyplot()

# Regional Price Comparison for all commodities
st.subheader('Regional Price Comparison for All Commodities')
average_prices = data.groupby('States/UTs')[commodities].mean()

fig, ax = plt.subplots(figsize=(12, 8))
average_prices.plot(kind='bar', stacked=False, ax=ax, figsize=(15, 8))
plt.title("Average Prices of Commodities Across States/UTs")
plt.xlabel('States/UTs')
plt.ylabel('Average Price (INR)')
st.pyplot()

# Statistical Summary for all commodities
st.subheader('Statistical Summary for All Commodities')
stat_summary = data[commodities].describe().transpose()
st.write(stat_summary)
