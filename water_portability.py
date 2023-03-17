import pandas as pd
from sklearn.preprocessing import StandardScaler

# Load data from CSV
data_set = pd.read_csv('water_portability.csv')

# Split features and labels
X = data_set.drop('Potability', axis=1)
y = data_set['Potability']

# Check for missing values
print(X.isnull().sum())

# Impute missing values with mean
X = X.fillna(X.mean())


# Standardize features
scaler = StandardScaler()
X = scaler.fit_transform(X)