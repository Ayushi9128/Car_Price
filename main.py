from flask import Flask, render_template, request
import pandas as pd
import pickle
import numpy as np

app = Flask(__name__)

model = pickle.load(open("LinearRegression_Model.pkl", "rb"))
car = pd.read_csv("CleanedCar.csv")

@app.route('/')
def index():
    companies = sorted(car["company"].unique())
    car_models = sorted(car['name'].unique())
    year = sorted(car['year'].unique(), reverse=True)
    fuel_type = car['fuel_type'].unique()


    return render_template("index.html", companies=companies, car_models=car_models, years=year, fuel_types=fuel_type)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        company = request.form.get('company')
        car_model = request.form.get('car_model')
        year = request.form.get('year')
        fuel_type = request.form.get('fuel_type')
        kms_driven = request.form.get('kilo_driven')

        # Check if any field is missing
        if not all([company, car_model, year, fuel_type, kms_driven]):
            return "Please fill out all the fields!"

        # Validate numeric inputs
        year = int(year)
        kms_driven = int(kms_driven)
        if kms_driven < 0:
            return "Kilometers driven cannot be negative."

        # Make prediction
        prediction = model.predict(pd.DataFrame([[car_model, company, year, kms_driven, fuel_type]],
                                                columns=['name', 'company', 'year', 'kms_driven', 'fuel_type']))
        if prediction<=0:
            return "The car does not have any resale value"
        return "Prediction: Rs. " + str(np.round(prediction[0], 2))

    except ValueError:
        return "Year and Kilometers must be valid integers."
    except Exception as e:
        return f"Something went wrong: {str(e)}"

if __name__ == "__main__":
    app.run(debug=True)

