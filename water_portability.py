import pandas as pd
from sklearn.preprocessing import StandardScaler
# imports for model creation
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Load data from CSV
data_set = pd.read_csv("D:\Work_Project\water_portability_prediction_model\water_potability.csv")

# Split features and labels
X = data_set.drop('Potability', axis=1)
y = data_set['Potability']

# Check for missing values
print(X.isnull().sum())

# impute missing values with mean
X = X.fillna(X.mean())


# standardizing the features of the data set
scaler = StandardScaler()
X = scaler.fit_transform(X)

# at this point the data preprocessing is done. we now create the model using Random Forest Algorithm

# split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42)

# training the model using the training data set with the RandomForest Classifier method
random_forest_classifier = RandomForestClassifier(random_state=42)

# fitting the model
random_forest_classifier.fit(X_train, y_train)

# evaluating the model's performance
print('Training Accuracy:', random_forest_classifier.score(X_train, y_train))
print('Testing Accuracy:', random_forest_classifier.score(X_test, y_test))