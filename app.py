from flask import Flask, request, render_template
import joblib
import numpy as np

app = Flask(_name_)
model = joblib.load("model.pkl")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    data = [float(x) for x in request.form.values()]
    prediction = model.predict([data])[0]
    return render_template("index.html", prediction_text=f"Churn Prediction: {'Yes' if prediction==1 else 'No'}")

if _name_ == "_main_":
    app.run(debug=True)