import pandas as pd
from sklearn.preprocessing import StandardScaler
# imports for model creation
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

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

# at this point the data preprocessing is done. we now create the model using Random Forest Algorithm

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
