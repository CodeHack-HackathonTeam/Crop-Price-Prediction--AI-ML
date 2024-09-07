import pandas as pd

# Load the dataset
file_path = 'june.csv'
data = pd.read_csv(file_path)

# Melt the dataframe to reshape it into long format
# This will result in columns: 'DATE', 'States/UTs', 'Commodity', and 'Price'
melted_data = pd.melt(data, id_vars=['DATE', 'States/UTs'], 
                      var_name='Commodity', value_name='Price')

# Rename columns for clarity
melted_data.rename(columns={'DATE': 'Date', 'States/UTs': 'State/UT'}, inplace=True)

melted_data.to_csv("test.csv")
