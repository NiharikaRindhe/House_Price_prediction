# # #libraries needed flask , scikit-learn, pandas, pickle-mixin
# Required libraries
import pandas as pd
from flask import Flask, render_template, request
import pickle
import numpy as np

# Flask App Initialization
app = Flask(__name__)

# Load Data and Model
data = pd.read_csv('Cleaned_data.csv')  # Ensure the file path is correct
pipe = pickle.load(open("RidgeModel.pkl", 'rb'))


@app.route('/')
def index():
    # Extract unique locations
    locations = sorted(data['location'].unique())
    return render_template('index.html', locations=locations)


@app.route('/predict', methods=['POST'])
def predict():
    # Get input values from the form
    location = request.form.get('location')
    bhk = request.form.get('bhk')
    bath = request.form.get('bath')
    sqft = request.form.get('total_sqft')

    print(location, bhk, bath, sqft)

    # Create a DataFrame for the input
    input_df = pd.DataFrame([[location, sqft, bath, bhk]],
                            columns=['location', 'total_sqft', 'bath', 'bhk'])

    # Make the prediction
    prediction = pipe.predict(input_df)[0] * 1e5

    # Return the prediction as a response
    return str(np.round(prediction, 2))


if __name__ == '__main__':
    app.run(debug=True, port=5001)
