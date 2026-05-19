from flask import Flask, request, render_template
import pandas as pd
import joblib
import numpy as np

app = Flask(__name__)

model = joblib.load("road_accident_model.pkl")

model_columns = joblib.load("model_columns.pkl")


@app.route("/")
def home():

    return render_template("index1.html")


@app.route("/predict", methods=["POST"])
def predict():

    road_type = request.form.get("road_type")

    num_lanes = float(request.form.get("num_lanes"))

    curvature = request.form.get("curvature")

    speed_limit = float(request.form.get("speed_limit"))

    lighting = request.form.get("lighting")

    weather = request.form.get("weather")

    road_signs_present = request.form.get(
        "road_signs_present"
    )

    public_road = request.form.get("public_road")

    time_of_day = request.form.get("time_of_day")

    holiday = request.form.get("holiday")

    school_season = request.form.get(
        "school_season"
    )

    num_reported_accidents = float(
        request.form.get("num_reported_accidents")
    )


    input_data = pd.DataFrame({

        "road_type": [road_type],

        "num_lanes": [num_lanes],

        "curvature": [curvature],

        "speed_limit": [speed_limit],

        "lighting": [lighting],

        "weather": [weather],

        "road_signs_present": [
            road_signs_present
        ],

        "public_road": [public_road],

        "time_of_day": [time_of_day],

        "holiday": [holiday],

        "school_season": [school_season],

        "num_reported_accidents": [
            num_reported_accidents
        ]

    })


    input_data = pd.get_dummies(input_data)

    input_data = input_data.reindex(
        columns=model_columns,
        fill_value=0
    )


    prediction = model.predict(input_data)[0]


    if prediction < 0.2:

        risk = "Low Risk"

        color = "green"

        emoji = "✅"

        advice = "Road conditions appear safe."

    elif prediction < 0.5:

        risk = "Medium Risk"

        color = "orange"

        emoji = "⚠️"

        advice = "Drive carefully and follow traffic rules."

    else:

        risk = "High Risk"

        color = "red"

        emoji = "🚨"

        advice = "High accident possibility. Avoid overspeeding."


    risk_score = round(prediction * 100, 2)


    return render_template(

        "index1.html",

        prediction_text=risk,

        risk_score=risk_score,

        risk_color=color,

        emoji=emoji,

        advice=advice

    )


@app.route("/about")
def about():

    return """

    <h1>About Project</h1>

    <p>

    This Road Accident Risk Prediction system uses

    Machine Learning to estimate accident risk

    based on road and environmental conditions.

    </p>

    """


@app.route("/contact")
def contact():

    return """

    <h1>Contact</h1>

    <p>Email: project@example.com</p>

    """


if __name__ == "__main__":

    app.run(debug=True)