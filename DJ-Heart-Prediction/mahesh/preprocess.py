import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

# Load the data
file_path ='./HeartDiseaseTrain-Test.csv'
data = pd.read_csv(file_path)

# Step 1: Inspect the data
print(data.head())
print(data.info())
print(data.describe())

# Step 2: Check for missing values
print("Missing values:\n", data.isnull().sum())

# Step 3: Handle missing values (example: impute numerical data with mean)
num_cols = data.select_dtypes(include=['float64', 'int64']).columns
cat_cols = data.select_dtypes(include=['object']).columns

imputer = SimpleImputer(strategy='mean')
data[num_cols] = imputer.fit_transform(data[num_cols])

# Step 4: Encode categorical variables
data = pd.get_dummies(data, columns=cat_cols, drop_first=True)

# Step 5: Normalize numerical features
scaler = StandardScaler()
data[num_cols] = scaler.fit_transform(data[num_cols])

# Step 6: Split data into train/test if required
train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)

# Save preprocessed files if needed
train_data.to_csv('./Preprocessed_Train.csv', index=False)
test_data.to_csv('./Preprocessed_Test.csv', index=False)

print("Preprocessing completed. Train and test datasets saved.")
