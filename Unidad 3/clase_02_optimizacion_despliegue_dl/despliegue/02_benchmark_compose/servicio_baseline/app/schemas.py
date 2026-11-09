"""Esquemas de entrada/salida de la API, validados con Pydantic."""
from typing import List

from pydantic import BaseModel, Field


class PredictionRequest(BaseModel):
    """Imagen en escala de grises de 28x28 (valores en [0, 255] o en [0, 1])."""
    imagen: List[List[float]] = Field(
        ...,
        description="Matriz 28x28 de intensidades de píxel",
        min_length=28,
        max_length=28,
    )


class PredictionResponse(BaseModel):
    clase_id: int
    clase_nombre: str
    confianza: float
    latencia_ms: float


class HealthResponse(BaseModel):
    status: str
    modelo_cargado: bool
    version_modelo: str
