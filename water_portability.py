import pandas as pd
from sklearn.preprocessing import StandardScaler
# imports for model creation
from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import GridSearchCV, train_test_split

# soad data from CSV
data_set = pd.read_csv("D:\Work_Project\water_portability_prediction_model\water_potability.csv")
# DATA PREPROCESSING
# split features and labels
X = data_set.drop('Potability', axis=1)
y = data_set['Potability']

# Check for missing values
print(X.isnull().sum())

# impute missing values with mean
X = X.fillna(X.mean())

# FEATURE SELECTION
# standardizing the features of the data set
scaler = StandardScaler()
X = scaler.fit_transform(X)

# at this point the data preprocessing is done. we now create the model using Random Forest Algorithm

# MODEL TRAINING
# split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42)

# training the model using the training data set with the RandomForest Classifier method
random_forest_classifier = RandomForestClassifier(random_state=42)

# MODEL FITTING
# fitting the model
random_forest_classifier.fit(X_train, y_train)

# MODEL EVALUATION
# evaluating the model's performance
# printing the training and testing accuracy
print('Training Accuracy:', random_forest_classifier.score(X_train, y_train))
print('Testing Accuracy:', random_forest_classifier.score(X_test, y_test))

# OPTIMIZING THE PREDICTION MODEL
# using cross-validation and hyperparameterization

# split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(data_set.iloc[:, :-1], data_set.iloc[:, -1], test_size=0.15, random_state=42)

# Define the parameter grid for the grid search
param_grid = {
    'n_estimators': [100, 200, 500],
    'max_depth': [5, 10, 20, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}