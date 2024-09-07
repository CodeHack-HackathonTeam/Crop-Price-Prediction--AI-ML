import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Step 1: Load the dataset
data = pd.read_csv('test.csv')

# Step 2: Convert 'DATE' column to datetime format
data['Date'] = pd.to_datetime(data['Date'], format='%d/%m/%Y')

# Step 3: Handle missing values

data.fillna(data.median(), inplace=True)

# Step 4: Encode 'States/UTs' using label encoding
label_encoder = LabelEncoder()
data['States/UTs'] = label_encoder.fit_transform(data['States/UTs'])

# Step 5: Handle outliers (Optional)
# You can apply any method you prefer to detect and handle outliers, such as capping values

# Step 6: Feature scaling (Optional)
# If needed, you can apply scaling like MinMaxScaler or StandardScaler to normalize data

# Check the cleaned data
#feature engineering

data['Month'] = data['DATE'].dt.month
data['Day_of_Week'] = data['DATE'].dt.day_name()  # This will give you the day name (e.g., 'Monday')
data['Week_of_Year'] = data['DATE'].dt.isocalendar().week
data['Day_of_Month'] = data['DATE'].dt.day


print(data.head())


data.to_csv("karam.csv")