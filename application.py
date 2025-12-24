import pickle
import numpy as np
from flask import Flask, request, render_template

app = Flask(__name__)

# Load model and scaler
ridge_model = pickle.load(open("models/ridge.pkl", "rb"))
standard_scaler = pickle.load(open("models/scaler.pkl", "rb"))

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predictdata", methods=["GET", "POST"])
def predict_data():

    if request.method == "POST":
        Temperature = float(request.form["Temperature"])
        RH = float(request.form["RH"])
        Ws = float(request.form["Ws"])
        Rain = float(request.form["Rain"])
        FFMC = float(request.form["FFMC"])
        DMC = float(request.form["DMC"])
        ISI = float(request.form["ISI"])
        Classes = float(request.form["Classes"])
        Region = float(request.form["Region"])

        input_data = [[
            Temperature, RH, Ws, Rain,
            FFMC, DMC, ISI, Classes, Region
        ]]

        scaled_data = standard_scaler.transform(input_data)
        prediction = ridge_model.predict(scaled_data)

        return render_template(
            "home.html",
            results=round(prediction[0], 2)
        )

    return render_template("home.html")

if __name__ == "__main__":
    app.run(debug=True)
