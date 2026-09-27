from flask import Flask, render_template, request, jsonify
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier

app = Flask(__name__)

DATASET = "irrigation_prediction.csv"

# Load the real project dataset
df = pd.read_csv(DATASET)

TARGET = "Irrigation_Need"

FEATURES = [
    "Soil_Type", "Soil_pH", "Soil_Moisture", "Organic_Carbon",
    "Electrical_Conductivity", "Temperature_C", "Humidity",
    "Rainfall_mm", "Sunlight_Hours", "Wind_Speed_kmh",
    "Crop_Type", "Crop_Growth_Stage", "Season", "Irrigation_Type",
    "Water_Source", "Field_Area_hectare", "Mulching_Used",
    "Previous_Irrigation_mm", "Region"
]

NUMERIC = [
    "Soil_pH", "Soil_Moisture", "Organic_Carbon",
    "Electrical_Conductivity", "Temperature_C", "Humidity",
    "Rainfall_mm", "Sunlight_Hours", "Wind_Speed_kmh",
    "Field_Area_hectare", "Previous_Irrigation_mm"
]

CATEGORICAL = [c for c in FEATURES if c not in NUMERIC]

numeric_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="median"))
])

categorical_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("num", numeric_pipe, NUMERIC),
    ("cat", categorical_pipe, CATEGORICAL)
])

model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        class_weight="balanced"
    ))
])

X = df[FEATURES]
y = df[TARGET]
model.fit(X, y)

# Dropdown values come from the actual dataset
options = {
    col: sorted(df[col].dropna().astype(str).unique().tolist())
    for col in CATEGORICAL
}

def make_record(form):
    record = {}
    for col in FEATURES:
        if col in NUMERIC:
            record[col] = float(form[col])
        else:
            record[col] = form[col]
    return record

@app.route("/")
def home():
    return render_template(
        "index.html",
        options=options,
        dataset_rows=len(df),
        classes=sorted(y.astype(str).unique().tolist())
    )

@app.route("/predict", methods=["POST"])
def predict():
    try:
        record = make_record(request.form)
        input_df = pd.DataFrame([record])
        prediction = str(model.predict(input_df)[0])
        probabilities = model.predict_proba(input_df)[0]
        classes = model.classes_

        confidence = float(max(probabilities) * 100)

        return jsonify({
            "prediction": prediction,
            "confidence": round(confidence, 2)
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    app.run(debug=True)
