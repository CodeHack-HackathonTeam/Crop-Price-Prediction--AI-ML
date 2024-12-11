import pandas as pd

# Load the dataset (replace with your actual file path)
data = pd.read_csv('july.csv')

# Print first few rows to inspect the data
print(data.head())

# Clean the date column by fixing any missing separator
# Replace any invalid date formats (like '03/072024') with valid ones (like '03/07/2024')
data['Dates'] = data['Dates'].astype(str)  # Ensure dates are strings
data['Dates'] = data['Dates'].replace(r'(\d{2})/(\d{2})(\d{4})', r'\1/\2/\3', regex=True)

# Now, convert the cleaned 'Dates' column to datetime format
data['Dates'] = pd.to_datetime(data['Dates'], format='%d/%m/%Y', errors='coerce')

# Check for any remaining missing values after conversion
print(data['Dates'].isna().sum())  # Check for any NaT values (invalid dates)

# Melt the dataset from wide format to long format
long_format_data = data.melt(id_vars=['Dates', 'States/UTs'], 
                             var_name='Commodity', 
                             value_name='Price')

# Handle missing values (you can choose different methods, here we use forward fill)
long_format_data.fillna(method='ffill', inplace=True)  # Forward fill missing values

# Set the 'Dates' column as the index for time-series analysis
long_format_data.set_index('Dates', inplace=True)

# Save the reshaped data to a new CSV file (you can adjust the filename as needed)
long_format_data.to_csv('july_commodity_prices_long.csv')

# Optional: Display the first few rows of the transformed data
print(long_format_data.head())
