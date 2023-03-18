from flask import Flask, request, jsonify
import joblib

# loading the saved model from the directory
loaded_model = joblib.load('best_model.joblib')

# initializing the flask application
app = Flask(__name__)

# defining the method  that performs prediction using the loaded model and the data received from the application
@app.route('/predict', methods=['POST'])
def predict():
    # get the input data from the request
    input_data = request.get_json()
    
    # preprocessing the input data 
    # load the input data into a data frame
    data_frame = pd.DataFrame(input_data)

    # extract the target variable
    target_var = data_frame['Potability'].values

    # drop the target variable from the data frame
    data_frame.drop(['Potability'], axis=1, inplace=True)

    # normalize numerical variables
    scaler = StandardScaler()
    data_frame = scaler.fit_transform(data_frame)

    # combine the target variable and preprocessed features into a numpy array
    preprocessed_data = np.column_stack((data_frame, target_var))
    
    # make predictions using the loaded model
    predictions = loaded_model.predict(preprocessed_data)
    
    # return the predictions as a JSON response
    return jsonify(predictions.tolist())

# starting our flask application
if __name__ == '__main__':
    app.run(debug=True)

# after starting our flask application we can use curl to test the prediction
# curl -X POST -H "Content-Type: application/json" -d '{"feature_1": 0.1, "feature_2": 0.2, "feature_3": 0.3}' http://localhost:5000/predict

