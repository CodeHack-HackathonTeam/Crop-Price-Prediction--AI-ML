import pandas as pd


file_path = "june.csv"  # Replace with your actual file path
data = pd.read_csv(file_path)

# Display the first few rows to understand the structure
print(data.head())

# Check for missing values
print(data.isnull().sum())

# Forward fill and backward fill to handle missing values
data.fillna(method='ffill', inplace=True)
data.fillna(method='bfill', inplace=True)


data['DATE'] = pd.to_datetime(data['DATE'], format='%d/%m/%Y')

# Set 'DATE' as the index for time-series analysis
data.set_index('DATE', inplace=True)


# Rename columns to standardize names
data.rename(columns={
    'Tur/Arhar Dal': 'Arhar Dal',
    'Vanaspati (Packed)': 'Vanaspati',
    'Groundnut Oil (Packed)': 'Groundnut Oil',
    # Add more renaming as required
}, inplace=True)

# Create lagged features (e.g., price 7 days ago, 30 days ago)
data['Potato_Lag_7'] = data['Potato'].shift(7)
data['Potato_Lag_30'] = data['Potato'].shift(30)

# Create rolling averages
data['Potato_Rolling_Avg_7'] = data['Potato'].rolling(window=7).mean()

# Repeat for other commodities (Onion, Tomato, etc.)
data['Onion_Lag_7'] = data['Onion'].shift(7)
data['Onion_Rolling_Avg_7'] = data['Onion'].rolling(window=7).mean()

# Save the preprocessed data to a new CSV
data.to_csv("preprocessed_data.csv")
