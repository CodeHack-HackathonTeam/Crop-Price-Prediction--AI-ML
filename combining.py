import pandas as pd

# Load the two datasets (replace with your actual file paths)
data1 = pd.read_csv('june_commodity_prices_long.csv')
data2 = pd.read_csv('july_commodity_prices_long.csv')

# Concatenate the datasets vertically (stack rows)
concatenated_data = pd.concat([data1, data2], ignore_index=True)

# Check the concatenated data
print(concatenated_data.head())

# Save the concatenated dataset to a new CSV file
concatenated_data.to_csv('concatenated_data.csv', index=False)
