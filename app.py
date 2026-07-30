from flask import Flask, request, render_template
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from src.components.predict_pipeline import CustomData, PredictPipeline

application = Flask(__name__)

app = application

# home page

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predictdata', methods=['GET', 'POST'])
def predict_datapoint():
    if request.method == 'GET':
        return render_template('index.html')
    else:
        data = CustomData(
            age=int(request.form.get('age')),
            gender=request.form.get('gender'),
            marital_status=request.form.get('marital_status'),
            salary=float(request.form.get('salary')),
            employment_type=request.form.get('employment_type'),
            region=request.form.get('region'),
            has_dependents=request.form.get('has_dependents'),
            tenure_years=float(request.form.get('tenure_years'))
        )
        pred_df = data.get_data_as_dataframe()
        predict_pipeline = PredictPipeline()
        prediction = predict_pipeline.predict(pred_df)

        result = "Will be Enrolling" if prediction[0] == 1 else "Won't be Enrolling"

        return render_template(
            "index.html",
            results=result
        )

if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=True) #use 127.0.0.1:5000