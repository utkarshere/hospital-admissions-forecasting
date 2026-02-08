from fastapi import FastAPI
import pandas as pd
import numpy as np
import joblib
from .preprocess import preprocess

app = FastAPI()

model = joblib.load("artifacts/model.joblib")
feature_cols = joblib.load("artifacts/feature_columns.pkl")
cat_maps = joblib.load("artifacts/category_maps.pkl")

@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(payload: dict):
    df = pd.DataFrame([payload])
    X = preprocess(df)
    for col, categories in cat_maps.items():
        if col in X.columns:
            X[col] = pd.Categorical(X[col], categories=categories).codes

    for col in feature_cols:
        if col not in X.columns:
            X[col] = np.nan

    
    X = X[feature_cols]
    X = X.fillna(0)

    pred = model.predict(X)[0]
    pred = max(1, round(pred))


    return {'Admissions': pred}
