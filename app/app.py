from flask import Flask, render_template, request
import joblib, json, numpy as np, os

app = Flask(__name__)

MODEL_DIR = os.environ.get("MODEL_DIR", "/model")

def load_artifacts():
    model  = joblib.load(f"{MODEL_DIR}/model.pkl")
    scaler = joblib.load(f"{MODEL_DIR}/scaler.pkl")
    with open(f"{MODEL_DIR}/metrics.json") as f:
        metrics = json.load(f)
    return model, scaler, metrics

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    fields = ["Pregnancies","Glucose","BloodPressure","SkinThickness",
              "Insulin","BMI","DiabetesPedigreeFunction","Age"]
    try:
        values = [float(request.form[f]) for f in fields]
    except (KeyError, ValueError):
        return render_template("index.html", error="Please fill all fields correctly.")

    model, scaler, metrics = load_artifacts()
    arr = np.array(values).reshape(1, -1)
    arr_scaled = scaler.transform(arr)
    pred  = model.predict(arr_scaled)[0]
    proba = model.predict_proba(arr_scaled)[0][1]

    result = {
        "prediction": "High Risk" if pred == 1 else "Low Risk",
        "confidence": round(proba * 100, 1),
        "metrics": metrics,
    }
    return render_template("result.html", result=result, values=dict(zip(fields, values)))

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)