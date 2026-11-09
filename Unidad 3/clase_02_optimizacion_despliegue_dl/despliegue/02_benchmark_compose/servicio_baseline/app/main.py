"""
API REST que sirve el modelo optimizado (.tflite).

    GET  /health   -> estado del servicio
    POST /predict  -> clasifica una imagen 28x28 (Fashion-MNIST)

Configuración por variables de entorno (ver .env.example):
    MODEL_PATH  ruta al .tflite
    LOG_LEVEL   DEBUG | INFO | WARNING ...
    API_KEY     si se define, /predict exige la cabecera X-API-Key

Ejecución local:  uvicorn app.main:app --reload     (docs en /docs)
"""
import logging
import os
from contextlib import asynccontextmanager

import numpy as np
from fastapi import Depends, FastAPI, Header, HTTPException

from app.model_utils import obtener_modelo
from app.schemas import HealthResponse, PredictionRequest, PredictionResponse

logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO").upper())
logger = logging.getLogger("inference-api")

_modelo = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Carga el modelo una sola vez, al arrancar (no en cada petición)."""
    global _modelo
    try:
        _modelo = obtener_modelo()
        logger.info("Modelo cargado desde %s", _modelo.model_path)
    except FileNotFoundError as exc:
        logger.warning("El modelo no está disponible: %s", exc)
        _modelo = None
    yield


app = FastAPI(
    title="Servicio de inferencia — Fashion-MNIST BASE (Keras .h5)",
    description="Sirve la CNN base sin optimizar (.h5).",
    version="1.0.0",
    lifespan=lifespan,
)


def verificar_api_key(x_api_key: str | None = Header(default=None)):
    """Si API_KEY está definida en el entorno, exige que coincida."""
    esperada = os.environ.get("API_KEY")
    if esperada and x_api_key != esperada:
        raise HTTPException(status_code=401, detail="API key inválida o ausente")


@app.get("/health", response_model=HealthResponse)
def health():
    return HealthResponse(
        status="ok",
        modelo_cargado=_modelo is not None,
        version_modelo=os.path.basename(_modelo.model_path) if _modelo else "no_cargado",
    )


@app.post("/predict", response_model=PredictionResponse, dependencies=[Depends(verificar_api_key)])
def predict(request: PredictionRequest):
    if _modelo is None:
        raise HTTPException(status_code=503, detail="El modelo no está cargado.")
    try:
        resultado = _modelo.predecir(np.array(request.imagen))
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    return PredictionResponse(**resultado)
