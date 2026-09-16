"""
FastAPI service to serve admission & malaria prediction models.
Run:
    uvicorn api.app:app --reload --port 8000
Docs:
    http://localhost:8000/docs
"""

import os
import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from api.schemas import AdmissionInput, MalariaInput, PredictionOut

MODEL_DIR = "models"
ADMISSION_MODEL = os.path.join(MODEL_DIR, "admission_best.pkl")
MALARIA_MODEL = os.path.join(MODEL_DIR, "malaria_best.pkl")

app = FastAPI(
    title="COREP ML API",
    description="REST API for admission & malaria risk prediction.",
    version="1.0.0",
)
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)

_models = {}


@app.on_event("startup")
def load_models():
    for name, path in [("admission", ADMISSION_MODEL), ("malaria", MALARIA_MODEL)]:
        if os.path.exists(path):
            _models[name] = joblib.load(path)
            print(f"✅ Loaded {name} model from {path}")
        else:
            print(f"⚠️  {name} model not found at {path}")


@app.get("/", tags=["health"])
def root():
    return {
        "service": "COREP ML API",
        "models_loaded": list(_models.keys()),
        "endpoints": ["/predict/admission", "/predict/malaria", "/health"],
    }


@app.get("/health", tags=["health"])
def health():
    return {"status": "ok", "models": list(_models.keys())}


@app.post("/predict/admission", response_model=PredictionOut, tags=["predict"])
def predict_admission(payload: AdmissionInput):
    model = _models.get("admission")
    if model is None:
        raise HTTPException(503, "Admission model not loaded. Train it first.")

    X = pd.DataFrame([payload.dict()])
    prob = float(model.predict_proba(X)[0, 1])
    pred = int(prob >= 0.5)
    label = "Likely Admitted" if pred else "Likely Outpatient"
    return PredictionOut(prediction=pred, probability=round(prob, 4),
                         label=label, model="admission_best")


@app.post("/predict/malaria", response_model=PredictionOut, tags=["predict"])
def predict_malaria(payload: MalariaInput):
    model = _models.get("malaria")
    if model is None:
        raise HTTPException(503, "Malaria model not loaded. Train it first.")

    X = pd.DataFrame([payload.dict()])
    prob = float(model.predict_proba(X)[0, 1])
    pred = int(prob >= 0.5)
    label = "Malaria Positive (likely)" if pred else "Malaria Negative (likely)"
    return PredictionOut(prediction=pred, probability=round(prob, 4),
                         label=label, model="malaria_best")


@app.get("/models", tags=["info"])
def list_models():
    return {"available": list(_models.keys())}