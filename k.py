import pandas as pd

# Load the datasets
price_data = pd.read_csv('test.csv')  # Adjust the file path as needed
weather_data = pd.read_csv('india_weather_august_2023_to_2024.csv')  # Adjust the file path as needed

# Print the column names to ensure correct names
print("Price Data Columns:", price_data.columns)
print("Weather Data Columns:", weather_data.columns)

# Check the first few rows of both datasets to understand the structure
print("Sample Price Data:\n", price_data.head())
print("Sample Weather Data:\n", weather_data.head())

# Ensure the 'Date' columns are formatted correctly
# Remove any leading/trailing spaces and specify format
price_data['Date'] = pd.to_datetime(price_data['Date'], format='%d/%m/%Y', errors='coerce')
weather_data['Date'] = pd.to_datetime(weather_data['Date'], errors='coerce')

# Remove any rows where date conversion failed
price_data = price_data.dropna(subset=['Date'])
weather_data = weather_data.dropna(subset=['Date'])

# Ensure 'State/UT' columns are formatted correctly
price_data['State/UT'] = price_data['State/UT'].str.strip()
weather_data['State/UT'] = weather_data['State/UT'].str.strip()

# Print unique values to check for discrepancies
print("Unique States in Price Data:", price_data['State/UT'].unique())
print("Unique States in Weather Data:", weather_data['State/UT'].unique())

# Merge datasets on 'State/UT' and 'Date'
try:
    combined_data = pd.merge(price_data, weather_data, on=['State/UT', 'Date'], how='left')
    combined_data.to_csv('combined_price_weatherv2.csv', index=False)
    print("Datasets merged and saved to 'combined_price_weatherv2.csv'.")
except KeyError as e:
    print(f"Error during merge: {e}")
