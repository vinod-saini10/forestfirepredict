from flask import Flask, request, jsonify, render_template
import numpy as np
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler

application = Flask(__name__)
app = application

## Load ridge regression model and scaler
ridge_model = pickle.load(open('models/ridge.pkl', 'rb'))
standard_scaler = pickle.load(open('models/scaler.pkl', 'rb'))

@app.route("/")
def index():
    return render_template('index.html')

@app.route('/predictdata', methods=['GET', 'POST'])
def predict_datapoint():
    if request.method == 'POST':
        try:
            Temperature = float(request.form['Temperature'])
            RH = float(request.form['RH'])
            Ws = float(request.form['Ws'])
            Rain = float(request.form['Rain'])
            FFMC = float(request.form['FFMC'])
            DMC = float(request.form['DMC'])
            ISI = float(request.form['ISI'])
            Classes = float(request.form['Classes'])
            Region = float(request.form['Region'])
        except (TypeError, ValueError, KeyError):
            return render_template('home.html', result="Invalid input. Please fill all fields with valid numbers.")

        new_data_scaled = standard_scaler.transform([[Temperature, RH, Ws, Rain, FFMC, DMC, ISI, Classes, Region]])
        result = ridge_model.predict(new_data_scaled)
        return render_template('home.html', result=result)
    else:
        return render_template('home.html')



if __name__ == "__main__":
    app.run(host="0.0.0.0",debug=True)
