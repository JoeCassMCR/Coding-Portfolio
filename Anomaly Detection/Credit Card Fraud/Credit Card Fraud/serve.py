"""
serve.py

FastAPI app to serve model predictions.
"""

import logging
import pickle
from fastapi import FastAPI
from pydantic import BaseModel, Field
import numpy as np

from typing import List
from preprocessing import preprocess_data
from config import DEFAULT_MODEL_PATH

app = FastAPI()
LOGGER = logging.getLogger(__name__)

# Load model at startup
with open(DEFAULT_MODEL_PATH, 'rb') as f:
    model = pickle.load(f)


class Transaction(BaseModel):
    V: List[float] = Field(..., min_items=28, max_items=28)
    log_amount: float
    hour_of_day: int


class PredictionResponse(BaseModel):
    probability: float
    is_fraud: bool


@app.post('/predict', response_model=PredictionResponse)
async def predict(transaction: Transaction):
    # Construct feature vector
    features = np.array([transaction.V + [transaction.log_amount, transaction.hour_of_day]])
    prob = model.predict_proba(features)[0, 1]
    return PredictionResponse(probability=prob, is_fraud=(prob >= 0.5))
