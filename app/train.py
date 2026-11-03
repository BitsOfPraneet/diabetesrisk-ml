import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
import joblib, os, json

DATA_PATH = os.environ.get("DATA_PATH", "/app/data/diabetes.csv")
MODEL_DIR = os.environ.get("MODEL_DIR", "/model")

def train():
    df = pd.read_csv(DATA_PATH)
    X = df.drop("Outcome", axis=1)
    y = df["Outcome"]

    # Replace 0s with median for physiologically invalid zeros
    cols_with_zeros = ["Glucose","BloodPressure","SkinThickness","Insulin","BMI"]
    for col in cols_with_zeros:
        df[col] = df[col].replace(0, df[col].median())

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train_scaled, y_train)

    y_pred = model.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)

    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(model,  f"{MODEL_DIR}/model.pkl")
    joblib.dump(scaler, f"{MODEL_DIR}/scaler.pkl")

    metrics = {
        "accuracy": round(acc * 100, 2),
        "precision_0": round(report["0"]["precision"], 2),
        "precision_1": round(report["1"]["precision"], 2),
        "recall_0":    round(report["0"]["recall"], 2),
        "recall_1":    round(report["1"]["recall"], 2),
        "f1_0":        round(report["0"]["f1-score"], 2),
        "f1_1":        round(report["1"]["f1-score"], 2),
    }
    with open(f"{MODEL_DIR}/metrics.json", "w") as f:
        json.dump(metrics, f)

    print(f"Training complete. Accuracy: {acc:.2%}")

if __name__ == "__main__":
    train()