from flask import Flask, render_template, request
import joblib
import json
import plotly.graph_objs as go
from plotly.utils import PlotlyJSONEncoder
import pandas as pd
import numpy as np

app = Flask(__name__)

# Load your trained Random Forest model
model = joblib.load('model/random_forest_classifier.pkl')

# Sample data to visualize in the dashboard (for demonstration)
predictions = {
    'Prediction 0': 0,
    'Prediction 1': 0
}

# Example feature names, replace these with your actual feature names
feature_names = ['Delivery Distance', 'Delivery Time', 'Delivery Urgency', 'Delivery Location']

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Extract features from the form
    feature1 = request.form['feature1']
    feature2 = request.form['feature2']
    feature3 = request.form['feature3']
    feature4 = request.form['feature4']
    
    input_data = [[float(feature1), float(feature2), float(feature3), float(feature4)]]
    
    # Make predictions
    prediction = model.predict(input_data)[0]

    # Store prediction result
    if prediction == 0:
        predictions['Prediction 0'] += 1
    else:
        predictions['Prediction 1'] += 1

    return render_template('index.html', prediction=prediction)

@app.route('/dashboard')
def dashboard():
    # Create a bar chart for the predictions
    data = [
        go.Bar(
            x=list(predictions.keys()),
            y=list(predictions.values()),
            marker=dict(color=['blue', 'red'])
        )
    ]

    layout = go.Layout(
        title='Prediction Results Overview',
        xaxis=dict(title='Predictions'),
        yaxis=dict(title='Count')
    )

    graphJSON = json.dumps(data, cls=PlotlyJSONEncoder)  # Use the correct encoder

    # Get feature importances from the Random Forest model
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]  # Sort feature importances in descending order

    # Create a feature importance chart
    importance_data = [
        go.Bar(
            x=[feature_names[i] for i in indices],
            y=importances[indices],
            marker=dict(color='green')
        )
    ]

    importance_layout = go.Layout(
        title='Feature Importances',
        xaxis=dict(title='Features'),
        yaxis=dict(title='Importance Score')
    )

    importance_graphJSON = json.dumps(importance_data, cls=PlotlyJSONEncoder)

    return render_template('dashboard.html', graphJSON=graphJSON, importance_graphJSON=importance_graphJSON)

if __name__ == '__main__':
    app.run(debug=True)
