from flask import Flask, request, render_template, redirect, url_for
import pickle
import pandas as pd
import numpy as np

# Custom transformer used in pipeline
def convert_age_column(X):
    def convert(age):
        if age == "unknown":
            return np.nan
        elif age == "Over 95":
            return 97
        else:
            try:
                start, end = map(int, age.split(" to "))
                return (start + end) / 2
            except:
                return np.nan
    return pd.DataFrame(X).map(convert)

# Reverse mapping of encoded predictions
reverse_injury_mapping = {0: "Minimal", 1: "Minor", 2: "Major", 3: "Fatal"}

# Load the model
with open("rf_pipeline_raw_input.pkl", "rb") as f:
    model = pickle.load(f)

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        # Collect form input
        input_data = {
            'INVAGE': request.form['INVAGE'],
            'ACCLASS': request.form['ACCLASS'],
            'DIVISION': request.form['DIVISION'],
            'IMPACTYPE': request.form['IMPACTYPE'],
            'INVTYPE': request.form['INVTYPE'],
            'LIGHT': request.form['LIGHT'],
            'INITDIR': request.form['INITDIR'],
            'TRAFFCTL': request.form['TRAFFCTL'],
            'ROAD_CLASS': request.form['ROAD_CLASS'],
            'DRIVACT': request.form['DRIVACT']
        }

        # Convert to DataFrame and store temporarily in session (optional)
        df = pd.DataFrame([input_data])
        prediction = model.predict(df)[0]
        predicted_label = reverse_injury_mapping.get(prediction, "Unknown")

        # Redirect to result page with prediction
        return redirect(url_for("result", pred=predicted_label))

    return render_template("form.html")


@app.route("/result")
def result():
    pred = request.args.get("pred")
    return render_template("result.html", result=pred)

if __name__ == "__main__":
    app.run(debug=True)
