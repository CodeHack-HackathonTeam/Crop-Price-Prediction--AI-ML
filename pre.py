import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Step 1: Load the dataset
data = pd.read_csv('test.csv')

# Step 2: Convert 'DATE' column to datetime format
data['Date'] = pd.to_datetime(data['Date'], format='%d/%m/%Y')

# Step 3: Handle missing values

data['Price'].fillna(data['Price'].median(), inplace=True)

# Step 4: Encode 'States/UTs' using label encoding
label_encoder = LabelEncoder()
data['State/UT'] = label_encoder.fit_transform(data['State/UT'])

# Step 5: Handle outliers (Optional)
# You can apply any method you prefer to detect and handle outliers, such as capping values

# Step 6: Feature scaling (Optional)
# If needed, you can apply scaling like MinMaxScaler or StandardScaler to normalize data

# Check the cleaned data
#feature engineering

data['Month'] = data['Date'].dt.month
data['Day_of_Week'] = data['Date'].dt.day_name()  # This will give you the day name (e.g., 'Monday')
data['Week_of_Year'] = data['Date'].dt.isocalendar().week
data['Day_of_Month'] = data['Date'].dt.day


print(data.head())


data.to_csv("karam.csv")