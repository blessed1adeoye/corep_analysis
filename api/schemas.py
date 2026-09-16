from pydantic import BaseModel, Field
from typing import Optional, Literal


class AdmissionInput(BaseModel):
    Age: float = Field(..., ge=0, le=120, example=34)
    Gender: Literal["Male", "Female", "Other"] = "Male"
    n_consultations: int = Field(1, ge=0, example=3)
    avg_sys: Optional[float] = 120
    avg_dia: Optional[float] = 80
    avg_temp: Optional[float] = 36.8
    avg_pulse: Optional[float] = 78
    avg_spo2: Optional[float] = 97


class MalariaInput(BaseModel):
    Age: float = Field(..., ge=0, le=120, example=12)
    Gender: Literal["Male", "Female", "Other"] = "Female"
    avg_temp: Optional[float] = 38.5
    avg_pulse: Optional[float] = 95


class PredictionOut(BaseModel):
    prediction: int
    probability: float
    label: str
    model: str

    