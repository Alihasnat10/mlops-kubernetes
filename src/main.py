# SPDX-License-Identifier: MIT
from __future__ import annotations
import os
from fastapi import FastAPI
from imdb_sentiment.inference import SentimentModel
from imdb_sentiment.schemas import PredictRequest, PredictResponse, PredictResponseItem

APP_NAME = os.environ.get("APP_NAME", "imdb-sentiment-api")
MODEL_DIR = os.environ.get("MODEL_DIR", "/models/model")

app = FastAPI(title=APP_NAME)
model: SentimentModel | None = None


@app.on_event("startup")
async def _load_model():
    global model
    model = SentimentModel(MODEL_DIR)


@app.get("/health")
async def health():
    return {"status": "ok", "model_dir": MODEL_DIR}


@app.post("/predict", response_model=PredictResponse)
async def predict(req: PredictRequest) -> PredictResponse:
    assert model is not None, "Model not loaded"
    preds = model.predict(req.texts)
    return PredictResponse(results=[PredictResponseItem(**p) for p in preds])

