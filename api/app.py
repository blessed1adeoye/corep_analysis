# """
# FastAPI service to serve admission & malaria prediction models.
# Run:
#     uvicorn api.app:app --reload --port 8000
# Docs:
#     http://localhost:8000/docs
# """

# import os
# import joblib
# import numpy as np
# import pandas as pd
# from fastapi import FastAPI, HTTPException
# from fastapi.middleware.cors import CORSMiddleware

# from api.schemas import AdmissionInput, MalariaInput, PredictionOut

# MODEL_DIR = "models"
# ADMISSION_MODEL = os.path.join(MODEL_DIR, "admission_best.pkl")
# MALARIA_MODEL = os.path.join(MODEL_DIR, "malaria_best.pkl")

# app = FastAPI(
#     title="COREP ML API",
#     description="REST API for admission & malaria risk prediction.",
#     version="1.0.0",
# )
# app.add_middleware(
#     CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
# )

# _models = {}


# @app.on_event("startup")
# def load_models():
#     for name, path in [("admission", ADMISSION_MODEL), ("malaria", MALARIA_MODEL)]:
#         if os.path.exists(path):
#             _models[name] = joblib.load(path)
#             print(f"✅ Loaded {name} model from {path}")
#         else:
#             print(f"⚠️  {name} model not found at {path}")


# @app.get("/", tags=["health"])
# def root():
#     return {
#         "service": "COREP ML API",
#         "models_loaded": list(_models.keys()),
#         "endpoints": ["/predict/admission", "/predict/malaria", "/health"],
#     }


# @app.get("/health", tags=["health"])
# def health():
#     return {"status": "ok", "models": list(_models.keys())}


# @app.post("/predict/admission", response_model=PredictionOut, tags=["predict"])
# def predict_admission(payload: AdmissionInput):
#     model = _models.get("admission")
#     if model is None:
#         raise HTTPException(503, "Admission model not loaded. Train it first.")

#     X = pd.DataFrame([payload.dict()])
#     prob = float(model.predict_proba(X)[0, 1])
#     pred = int(prob >= 0.5)
#     label = "Likely Admitted" if pred else "Likely Outpatient"
#     return PredictionOut(prediction=pred, probability=round(prob, 4),
#                          label=label, model="admission_best")


# @app.post("/predict/malaria", response_model=PredictionOut, tags=["predict"])
# def predict_malaria(payload: MalariaInput):
#     model = _models.get("malaria")
#     if model is None:
#         raise HTTPException(503, "Malaria model not loaded. Train it first.")

#     X = pd.DataFrame([payload.dict()])
#     prob = float(model.predict_proba(X)[0, 1])
#     pred = int(prob >= 0.5)
#     label = "Malaria Positive (likely)" if pred else "Malaria Negative (likely)"
#     return PredictionOut(prediction=pred, probability=round(prob, 4),
#                          label=label, model="malaria_best")


# @app.get("/models", tags=["info"])
# def list_models():
#     return {"available": list(_models.keys())}

"""
COREP ML API — FastAPI service for admission & malaria risk prediction.

Run:
    uvicorn api.app:app --reload --port 8000

Docs:
    http://localhost:8000/docs

Features:
  - Loads trained models from ./models/ if available
  - Falls back to heuristic rules if models are missing
  - Full OpenAPI documentation
"""

import os
import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from api.schemas import (
    AdmissionInput, MalariaInput, PredictionOut, HealthOut,
)

# ============================================================
# Config
# ============================================================
MODEL_DIR = "models"
ADMISSION_MODEL = os.path.join(MODEL_DIR, "admission_best.pkl")
MALARIA_MODEL = os.path.join(MODEL_DIR, "malaria_best.pkl")

VERSION = "1.0.0"

app = FastAPI(
    title="COREP ML API",
    description=(
        "REST API for admission & malaria risk prediction. "
        "Uses trained models when available; falls back to heuristic rules otherwise."
    ),
    version=VERSION,
    contact={
        "name": "COREP Analytics",
        "url": "https://corep-dashboard.streamlit.app",
    },
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],       # tighten for production
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================
# Model loading
# ============================================================
_models = {}


@app.on_event("startup")
def load_models():
    """Attempt to load trained models; log which are available."""
    print("\n" + "=" * 60)
    print(" COREP ML API — startup")
    print("=" * 60)

    for name, path in [
        ("admission", ADMISSION_MODEL),
        ("malaria", MALARIA_MODEL),
    ]:
        if os.path.exists(path):
            try:
                _models[name] = joblib.load(path)
                print(f"✅ Loaded {name:10s} ← {path}")
            except Exception as e:
                print(f"⚠️  {name:10s} failed to load: {e}")
        else:
            print(f"ℹ️  {name:10s} not found at {path} — will use heuristic fallback")

    print(f"\n✅ API ready. Models loaded: {list(_models.keys()) or ['none (heuristic mode)']}")


# ============================================================
# Heuristic fallbacks
# ============================================================
def _heuristic_admission(payload: AdmissionInput) -> tuple:
    """Rule-based admission risk score."""
    score = 0

    # Age
    if payload.Age >= 65: score += 3
    elif payload.Age >= 50: score += 2
    elif payload.Age <= 5: score += 2

    # Vitals
    if payload.avg_sys and payload.avg_sys >= 160: score += 3
    elif payload.avg_sys and payload.avg_sys >= 140: score += 2
    if payload.avg_temp and payload.avg_temp >= 39: score += 2
    if payload.avg_spo2 and payload.avg_spo2 <= 92: score += 3
    elif payload.avg_spo2 and payload.avg_spo2 <= 94: score += 1
    if payload.avg_pulse and payload.avg_pulse >= 110: score += 2

    # Consultation count
    if payload.n_consultations >= 4: score += 2

    # Convert to probability (0–1)
    prob = min(0.95, score / 14)
    pred = int(prob >= 0.5)
    return pred, round(prob, 4)


def _heuristic_malaria(payload: MalariaInput) -> tuple:
    """Rule-based malaria risk score."""
    score = 0

    # Age (children and young adults most affected)
    if payload.Age <= 12: score += 3
    elif payload.Age <= 35: score += 2

    # Fever is the strongest signal
    if payload.avg_temp and payload.avg_temp >= 39: score += 4
    elif payload.avg_temp and payload.avg_temp >= 38: score += 3
    elif payload.avg_temp and payload.avg_temp >= 37.5: score += 2

    # Tachycardia
    if payload.avg_pulse and payload.avg_pulse >= 110: score += 2
    elif payload.avg_pulse and payload.avg_pulse >= 95: score += 1

    # Hypotension
    if payload.avg_sys and payload.avg_sys < 100: score += 2

    prob = min(0.95, score / 13)
    pred = int(prob >= 0.5)
    return pred, round(prob, 4)


# ============================================================
# Endpoints
# ============================================================
@app.get("/", tags=["health"])
def root():
    return {
        "service": "COREP ML API",
        "version": VERSION,
        "models_loaded": list(_models.keys()) or ["heuristic mode"],
        "endpoints": [
            "/predict/admission",
            "/predict/malaria",
            "/health",
            "/docs",
        ],
    }


@app.get("/health", response_model=HealthOut, tags=["health"])
def health():
    return HealthOut(
        status="ok",
        models_loaded=list(_models.keys()) or ["heuristic"],
        version=VERSION,
    )


@app.post("/predict/admission", response_model=PredictionOut, tags=["predict"])
def predict_admission(payload: AdmissionInput):
    """
    Predict likelihood of hospital admission.

    Uses trained model if available, else heuristic rules.
    """
    if "admission" in _models:
        try:
            X = pd.DataFrame([payload.dict()])
            prob = float(_models["admission"].predict_proba(X)[0, 1])
            pred = int(prob >= 0.5)
            label = "Likely Admitted" if pred else "Likely Outpatient"
            return PredictionOut(
                prediction=pred, probability=round(prob, 4),
                label=label, model="admission_best", mode="trained",
            )
        except Exception as e:
            print(f"⚠️  Trained model failed, falling back: {e}")

    pred, prob = _heuristic_admission(payload)
    label = "Likely Admitted" if pred else "Likely Outpatient"
    return PredictionOut(
        prediction=pred, probability=prob,
        label=label, model="heuristic_rules", mode="heuristic",
    )


@app.post("/predict/malaria", response_model=PredictionOut, tags=["predict"])
def predict_malaria(payload: MalariaInput):
    """
    Predict likelihood of malaria positivity.

    Uses trained model if available, else heuristic rules.
    """
    if "malaria" in _models:
        try:
            X = pd.DataFrame([payload.dict()])
            prob = float(_models["malaria"].predict_proba(X)[0, 1])
            pred = int(prob >= 0.5)
            label = "Malaria Positive (likely)" if pred else "Malaria Negative (likely)"
            return PredictionOut(
                prediction=pred, probability=round(prob, 4),
                label=label, model="malaria_best", mode="trained",
            )
        except Exception as e:
            print(f"⚠️  Trained model failed, falling back: {e}")

    pred, prob = _heuristic_malaria(payload)
    label = "Malaria Positive (likely)" if pred else "Malaria Negative (likely)"
    return PredictionOut(
        prediction=pred, probability=prob,
        label=label, model="heuristic_rules", mode="heuristic",
    )


@app.get("/models", tags=["info"])
def list_models():
    return {
        "available": list(_models.keys()) or ["none"],
        "admission_path": ADMISSION_MODEL,
        "malaria_path": MALARIA_MODEL,
    }