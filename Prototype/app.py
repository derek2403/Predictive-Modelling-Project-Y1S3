import numpy as np
from flask import Flask, request, render_template, abort
import pickle
import logging

app = Flask(__name__)

logging.basicConfig(filename='app.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

model = pickle.load(open("model.pkl", "rb"))

feature_names = ['smv', 'wip', 'over_time', 'incentive', 'no_of_workers']

@app.route("/")
def Home():
    return render_template("index.html")

@app.route("/predict", methods=["POST", "GET"])
def predict():
    logging.info(f"Request method: {request.method}")

    if request.method == "POST":
        try:
            targeted_productivity = float(request.form['targeted_productivity'])
            if not 0 <= targeted_productivity <= 1:
                raise ValueError("Targeted productivity should be between 0 and 1")

            smv = float(request.form['smv'])
            over_time = float(request.form['over_time'])
            incentive = float(request.form['incentive'])
            no_of_workers = float(request.form['no_of_workers'])

            # Client-side validation for other input fields
            if not all(map(lambda x: x.replace('.', '', 1).isdigit(), [request.form['smv'], request.form['over_time'], request.form['incentive'], request.form['no_of_workers']])):
                raise ValueError("Please enter valid numerical values for SMV, Over Time, Incentive, and Number of Workers.")

            features = np.array([[targeted_productivity, smv, over_time, incentive, no_of_workers]])

            prediction = model.predict(features)

            return render_template("index.html", prediction=prediction[0])
        except ValueError as ve:
            logging.error(f"ValueError: {str(ve)}")
            return render_template("index.html", error=str(ve))
        except Exception as e:
            logging.error(f"Error: {str(e)}")
            return render_template("error.html", error="An error occurred while making the prediction.")

    elif request.method == "GET":
        return render_template("index.html")

    else:
        logging.warning(f"Unsupported request method: {request.method}")
        abort(405)

if __name__ == "__main__":
    app.run(debug=True)
