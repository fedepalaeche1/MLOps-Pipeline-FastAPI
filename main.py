"""
API de predicción de aprobación de crédito con FastAPI.

Carga el pipeline completo (preprocesamiento + modelo) guardado en pipeline.joblib
y expone un endpoint /predict que recibe datos crudos.

Ejecutar:
    uvicorn main:app --host 0.0.0.0 --port 8000
"""
from pathlib import Path
from typing import Literal, Optional

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

PIPELINE_PATH = Path(__file__).resolve().parent / "pipeline.joblib"
COLUMNS = ["edad", "ingreso_mensual", "deuda_actual", "antiguedad_laboral",
           "tipo_contrato", "ciudad", "nivel_educativo"]

model = joblib.load(PIPELINE_PATH)

app = FastAPI(
    title="API Aprobación de Crédito",
    description="Pipeline de scikit-learn (imputación + escalado + one-hot + regresión logística) servido con FastAPI.",
    version="1.0.0",
)


class Solicitud(BaseModel):
    edad: float = Field(..., ge=18, le=100, examples=[35])
    ingreso_mensual: Optional[float] = Field(None, ge=0, examples=[6000])
    deuda_actual: float = Field(..., ge=0, examples=[1000])
    antiguedad_laboral: Optional[float] = Field(None, ge=0, examples=[8])
    tipo_contrato: Optional[Literal["indefinido", "fijo", "independiente"]] = Field(None, examples=["indefinido"])
    ciudad: str = Field(..., examples=["Medellin"])
    nivel_educativo: Optional[Literal["bachiller", "tecnico", "profesional", "posgrado"]] = Field(None, examples=["profesional"])


class SolicitudLote(BaseModel):
    solicitudes: list[Solicitud] = Field(..., min_length=1)

    model_config = {
        "json_schema_extra": {
            "examples": [{
                "solicitudes": [
                    {"edad": 35, "ingreso_mensual": 6000, "deuda_actual": 1000, "antiguedad_laboral": 8,
                     "tipo_contrato": "indefinido", "ciudad": "Medellin", "nivel_educativo": "profesional"},
                    {"edad": 22, "ingreso_mensual": None, "deuda_actual": 9000, "antiguedad_laboral": 0,
                     "tipo_contrato": "independiente", "ciudad": "Pereira", "nivel_educativo": None},
                ]
            }]
        }
    }


@app.get("/")
def home():
    return {"Universidad EIA": "MLOps", "servicio": "API Aprobación de Crédito", "docs": "/docs"}


@app.get("/health")
def health():
    return {"status": "ok", "modelo": type(model.named_steps["model"]).__name__}


@app.post("/predict")
def predict(request: SolicitudLote):
    try:
        df = pd.DataFrame([s.model_dump() for s in request.solicitudes], columns=COLUMNS)
        predicciones = model.predict(df)
        probabilidades = model.predict_proba(df)[:, 1]
        return {
            "predicciones": [
                {"aprobado": int(p), "resultado": "APROBADO" if p == 1 else "RECHAZADO",
                 "prob_aprobado": round(float(prob), 4)}
                for p, prob in zip(predicciones, probabilidades)
            ]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
