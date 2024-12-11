import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the preprocessed data
data = pd.read_csv('preprocessed_data.csv', parse_dates=['DATE'])

# Set up Streamlit
st.title("Agri-Horticultural Price Prediction Dashboard")

# Sidebar for user input
st.sidebar.header('User Input Parameters')
selected_commodity = st.sidebar.selectbox("Select Commodity", ['All Commodities'] + list(data.columns[2:]))
selected_region = st.sidebar.selectbox("Select State/UT", data['States/UTs'].unique())

# 1. Display price trends for all commodities
st.subheader(f"Price Trend Over Time for All Commodities")
if selected_commodity == "All Commodities":
    commodities = data.columns[2:]  # Get all commodity columns
    for commodity in commodities:
        plt.figure(figsize=(10, 5))
        commodity_data = data[data['States/UTs'] == selected_region]
        plt.plot(commodity_data['DATE'], commodity_data[commodity], marker='o', linestyle='-', label=commodity)
    
    plt.title(f'Commodity Prices in {selected_region}')
    plt.xlabel('Date')
    plt.ylabel('Price (INR)')
    plt.legend(loc='best')
    st.pyplot()
else:
    # 2. Display price trend for the selected commodity
    commodity_data = data[data['States/UTs'] == selected_region]
    plt.figure(figsize=(10, 5))
    plt.plot(commodity_data['DATE'], commodity_data[selected_commodity], marker='o', linestyle='-', color='b')
    plt.title(f'{selected_commodity} Prices in {selected_region}')
    plt.xlabel('Date')
    plt.ylabel('Price (INR)')
    st.pyplot()

# 3. Show correlation matrix
st.subheader('Correlation Matrix of All Commodities')
commodities = data.columns[2:]  # List of all commodity columns
corr_matrix = data[commodities].corr()

fig, ax = plt.subplots(figsize=(12, 10))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", ax=ax)
st.pyplot()

# 4. Regional Price Comparison for all commodities
st.subheader('Regional Price Comparison for All Commodities')
average_prices = data.groupby('States/UTs')[commodities].mean()

fig, ax = plt.subplots(figsize=(12, 8))
average_prices.plot(kind='bar', stacked=False, ax=ax, figsize=(15, 8))
plt.title("Average Prices of Commodities Across States/UTs")
plt.xlabel('States/UTs')
plt.ylabel('Average Price (INR)')
st.pyplot()

# 5. Statistical Summary for all commodities
st.subheader('Statistical Summary for All Commodities')
stat_summary = data[commodities].describe().transpose()
st.write(stat_summary)
