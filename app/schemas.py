from pydantic import BaseModel
from typing import List

class Detection(BaseModel):
    class_name: str
    confidence: float
    bbox: List[float]
    bcrs_eligible: bool

class PredictionResponse(BaseModel):
    filename: str
    detections: List[Detection]
    total_detections: int
    eligible_count: int
    not_eligible_count: int