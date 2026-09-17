# from pydantic import BaseModel, Field
# from typing import Optional, Literal


# class AdmissionInput(BaseModel):
#     Age: float = Field(..., ge=0, le=120, example=34)
#     Gender: Literal["Male", "Female", "Other"] = "Male"
#     n_consultations: int = Field(1, ge=0, example=3)
#     avg_sys: Optional[float] = 120
#     avg_dia: Optional[float] = 80
#     avg_temp: Optional[float] = 36.8
#     avg_pulse: Optional[float] = 78
#     avg_spo2: Optional[float] = 97


# class MalariaInput(BaseModel):
#     Age: float = Field(..., ge=0, le=120, example=12)
#     Gender: Literal["Male", "Female", "Other"] = "Female"
#     avg_temp: Optional[float] = 38.5
#     avg_pulse: Optional[float] = 95


# class PredictionOut(BaseModel):
#     prediction: int
#     probability: float
#     label: str
#     model: str

    
"""
Pydantic schemas for the COREP ML API.
"""
from pydantic import BaseModel, Field
from typing import Optional, Literal


class AdmissionInput(BaseModel):
    """Input for admission risk prediction."""
    Age: float = Field(..., ge=0, le=120, example=34)
    Gender: Literal["Male", "Female", "Other"] = "Male"
    n_consultations: int = Field(1, ge=0, example=3)
    avg_sys: Optional[float] = Field(120, ge=0, le=300)
    avg_dia: Optional[float] = Field(80, ge=0, le=200)
    avg_temp: Optional[float] = Field(36.8, ge=30, le=45)
    avg_pulse: Optional[float] = Field(78, ge=0, le=250)
    avg_spo2: Optional[float] = Field(97, ge=0, le=100)


class MalariaInput(BaseModel):
    """Input for malaria risk prediction."""
    Age: float = Field(..., ge=0, le=120, example=12)
    Gender: Literal["Male", "Female", "Other"] = "Female"
    avg_temp: Optional[float] = Field(38.5, ge=30, le=45)
    avg_pulse: Optional[float] = Field(95, ge=0, le=250)
    avg_sys: Optional[float] = Field(110, ge=0, le=300)
    avg_dia: Optional[float] = Field(70, ge=0, le=200)


class PredictionOut(BaseModel):
    """Standard prediction response."""
    prediction: int
    probability: float
    label: str
    model: str
    mode: str = Field("trained", description="'trained' or 'heuristic'")


class HealthOut(BaseModel):
    status: str
    models_loaded: list
    version: str