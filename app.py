from flask import Flask, render_template, request
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import numpy as np

# Load the dataset
df = pd.read_csv('Advertising.csv')

# Split the data into features (X) and target variable (y)
X = df[['TV', 'Radio', 'Newspaper']]
y = df['Sales']

# Create a pipeline for the complete process
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('regressor', RandomForestRegressor())
])

# Fit the pipeline on the entire dataset
pipeline.fit(X, y)

# Define Flask application and route
app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        # Get input values from the form
        tv = float(request.form['tv'])
        radio = float(request.form['radio'])
        newspaper = float(request.form['newspaper'])

        # Create a DataFrame with the input values
        input_data = pd.DataFrame([[tv, radio, newspaper]], columns=['TV', 'Radio', 'Newspaper'])

        # Scale the input data
        input_scaled = pipeline.named_steps['scaler'].transform(input_data)

        # Make prediction
        prediction = pipeline.named_steps['regressor'].predict(input_scaled)

        return render_template('result.html', prediction=round(prediction[0],2))
    return render_template('form.html')

if __name__ == '__main__':
    app.run(debug=True)
