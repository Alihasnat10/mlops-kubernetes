from __future__ import annotations
from typing import List
from pydantic import BaseModel, Field


class PredictRequest(BaseModel):
    texts: List[str] = Field(..., description="List of input sentences for sentiment analysis")


class PredictResponseItem(BaseModel):
    label: str
    score: float
    all_scores: list[float]


class PredictResponse(BaseModel):
    results: List[PredictResponseItem]