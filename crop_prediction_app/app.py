from flask import Flask, render_template, request
import joblib
import numpy as np

# Load the trained model
model = joblib.load('crop_prediction_model1.pkl')

# Initialize Flask app
app = Flask(__name__)

# Define route for the home page
@app.route('/')
def home():
    return render_template('index.html')

# Define route for prediction
@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Get input values from the form
        N = float(request.form['N'])
        P = float(request.form['P'])
        K = float(request.form['K'])
        temperature = float(request.form['temperature'])
        humidity = float(request.form['humidity'])
        ph = float(request.form['ph'])
        rainfall = float(request.form['rainfall'])

        # Make prediction
        input_data = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
        prediction = model.predict(input_data)[0]

        return render_template('index.html', prediction=prediction)

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)
