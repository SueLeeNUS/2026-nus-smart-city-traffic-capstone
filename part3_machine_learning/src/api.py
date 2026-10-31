"""FastAPI mock: serves the production regressor. Thin wrapper; the logic lives in src/serving.py.

Run locally:  uvicorn src.api:app --reload
Sample request:
  curl -X POST http://127.0.0.1:8000/predict/volume -H "Content-Type: application/json" \
       -d '{"datetime": "2018-03-14 16:00:00", "temp": 280.0, "weather_main": "Clear"}'

NOTE: fastapi is not installed in the build sandbox, so this module has been syntax-checked only.
"""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from src.serving import load_regressor, predict_volume

app = FastAPI(title="Smart City Traffic Volume API", version="1.0.0")
_model = None


class PredictRequest(BaseModel):
    datetime: str
    temp: float
    weather_main: str = "Clear"
    rain_1h: float = 0.0
    snow_1h: float = 0.0
    clouds_all: int = 40
    holiday: str = "No Holiday"


@app.get("/health")
def health():
    return {"status": "ok", "model": "random_forest_regressor v1.0.0"}


@app.post("/predict/volume")
def predict(req: PredictRequest):
    global _model
    if _model is None:
        _model = load_regressor()
    try:
        return predict_volume(_model, req.model_dump())
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
