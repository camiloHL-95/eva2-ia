from fastapi import FastAPI, Query
from pydantic import BaseModel, Field
import pandas as pd
import joblib
import logging
from typing import Optional

app = FastAPI(title="Medical Models API — Diabetes & Insurance")

logger = logging.getLogger("medical_models_api")

#CARGA DE MODELOS
DIABETES_MODEL_PATH = "models/diabetes.pkl"
INSURANCE_MODEL_PATH = "models/insurance.pkl"

try:
    model_diabetes = joblib.load(DIABETES_MODEL_PATH)
except Exception as e:
    logger.exception("No se pudo cargar el modelo de diabetes")
    model_diabetes = None

try:
    model_insurance = joblib.load(INSURANCE_MODEL_PATH)
except Exception as e:
    logger.exception("No se pudo cargar el modelo de seguros")
    model_insurance = None

#ESQUEMAS DE ENTRADA
class DiabetesInput(BaseModel):
    Pregnancies: int
    Glucose: float
    BloodPressure: float
    SkinThickness: float
    Insulin: float
    BMI: float
    DiabetesPedigreeFunction: float
    Age: int

class InsuranceInput(BaseModel):
    age: int
    sex: str
    bmi: float
    children: int
    smoker: str
    region: str

@app.get("/")
def home():
    return {"message": "API de modelos (Diabetes y Seguros) funcionando 🚀"}

# ==== ENDPOINT DIABETES ====
@app.post("/predict/diabetes")
def predict_diabetes(
    payload: DiabetesInput,
    threshold: Optional[float] = Query(
        default=0.32, ge=0.0, le=1.0,
        description="Umbral de decisión para clasificar Outcome=1"
    )
):
    if model_diabetes is None:
        return {"error": "Modelo de diabetes no cargado"}

    df = pd.DataFrame([payload.dict()])
    proba_1 = float(model_diabetes.predict_proba(df)[:, 1][0])
    pred = int(proba_1 >= threshold)

    return {
        "probability": round(proba_1, 6),
        "threshold_used": threshold,
        "prediction": pred  # 1 = positivo (diabetes), 0 = negativo
    }

# ==== ENDPOINT INSURANCE (COSTOS) ====
@app.post("/predict/insurance")
def predict_insurance(payload: InsuranceInput):
    if model_insurance is None:
        return {"error": "Modelo de seguros no cargado"}

    df = pd.DataFrame([payload.dict()])
    cost = float(model_insurance.predict(df)[0])
    return {"predicted_cost": round(cost, 2)}
