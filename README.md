# Hospital Admissions Forecasting

An end-to-end **machine learning–based forecasting system** to predict daily hospital admissions using ensemble models, engineered time features, and grouped (panel) data across **Hospital × Department**, deployed as a public **FastAPI service**.

---

## 🔍 Problem Statement

Hospitals need accurate short-term forecasts of patient admissions to:

- plan staffing levels
- allocate resources
- manage operational load

Admissions are influenced by multiple factors such as:

- temporal patterns (day of week, seasonality)
- environmental conditions (temperature, air quality, precipitation)
- institutional factors (hospital, department)
- public events and holidays

This project models admissions as a **supervised learning problem with time-aware features.**

## 🧠 Approach

### Modeling Strategy

- **ML-based time series forecasting** using tabular data
- Tree-based ensemble models (Random Forest → LightGBM)
- Time encoded via:
  - lag features
  - rolling statistics
  - calendar features

---

### 🏗️ Feature Engineering

Key features include:

- **Calendar features**

  - day of week
  - week of year
  - month, quarter
  - weekend indicator
- **Lagged target features**

  - admissions_lag_1
  - admissions_lag_7
  - admissions_lag_14
- **Rolling statistics**

  - 7-day and 14-day rolling means of admissions
  - rolling averages of temperature, AQI, staffing
- **Environmental signals**

  - temperature
  - precipitation (log-transformed)
  - air quality index
- **Categorical variables**

  - Hospital_ID
  - Department
  - Flu_Activity (encoded numerically)

### 📊 Model

- **Final model**: LightGBM Regressor
- **Evaluation metric**: Mean Absolute Percentage Error (MAPE)
- **Validation performance**: ~10% MAPE

---

## 🚀 Deployment

The model is deployed as a **REST API** using:

- **FastAPI** (API framework)
- **Uvicorn** (ASGI server)
- **Render** (cloud hosting)

### 🔗 Live API

- Base URL:https://hospital-admissions-forecasting.onrender.com
- Swagger UI:
  https://hospital-admissions-forecasting.onrender.com/docs

## Project Structure

```css

HOSPITAL_ADMISSIONS/

│

├── admissions/ # Virtual environment (not tracked)

│ ├── etc/

│ ├── images/

│ ├── Include/

│ ├── Lib/

│ ├── Scripts/

│ ├── share/

│ ├── pyvenv.cfg

│ └── .gitignore

│

├── artifacts/ # Trained model & inference artifacts

│ ├── model.joblib # Final trained LightGBM model

│ ├── feature_columns.pkl # Feature order used during training

│ └── category_maps.pkl # Categorical encoding mappings

│

├── dataset/ # Raw datasets (ignored in Git)

│ ├── train.csv

│ ├── test.csv

│ └── sample_submission.csv

│

├── src/

│ ├── app/ # FastAPI application (production code)

│ │ ├── init.py

│ │ ├── main.py # API entry point

│ │ └── preprocess.py # Feature engineering & preprocessing

│ │

│ └── notebooks/ # Research & experimentation

│ ├── EDA.ipynb

│ └── modelling.ipynb

│

├── .gitignore

├── README.md

└── requirements.txt
```

## 🧪 Run Locally

### 1. Clone the repository

```bash
git clone https://github.com/utkarshere/hospital-admissions-forecasting.git
cd hospital_admissions-forecasting
```

### 2. Create and activate virtual environment

```
python -m venv admissions
source admissions/bin/activate   # Linux / Mac
admissions\Scripts\activate      # Windows

```

### 3. Install dependencies

```
pip install -r requirements.txt
```

### 4. Start the API server

```
uvicorn src.app.main:app --reload

```

### 5. Swagger UI

```
http://127.0.0.1:8000/docs
```

## Render Persistence/API Usage

```
https://hospital-admissions-forecasting.onrender.com/docs
```

### Endpoint

```
POST/predict
```

### Sample Request

```json
{
  "Date": "2024-01-15",
  "Hospital_ID": "H001",
  "Department": "Cardiology",
  "Temperature": 22.5,
  "Precipitation": 3.2,
  "Air_Quality_Index": 85,
  "Flu_Activity": "Moderate",
  "Staffing_Level": 110,
  "Public_Holiday": 0,
  "Weekend": 0,
  "Special_Events": 0
}
```

### Sample Response

```json

{

  "Admissions": 37

}
```

## Possible Improvements

- Probabilistic forecasting (prediction intervals)
- Hierarchical forecasting (hospital → department reconciliation)
- Online retraining pipeline
- Monitoring for data drift and performance decay
