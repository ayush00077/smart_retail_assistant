from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from app.services.azure_ml_service import predict_sales

router = APIRouter()

class ForecastRequest(BaseModel):
    store:        int
    date:         str       # "YYYY-MM-DD"
    holiday_flag: int       # 0 or 1
    temperature:  float
    fuel_price:   float
    cpi:          float
    unemployment: float
    day:          int
    month:        int
    year:         int

@router.post("/ml/predict")
async def forecast(payload: ForecastRequest):
    result = predict_sales(
        store=payload.store,
        date=payload.date,
        holiday_flag=payload.holiday_flag,
        temperature=payload.temperature,
        fuel_price=payload.fuel_price,
        cpi=payload.cpi,
        unemployment=payload.unemployment,
        day=payload.day,
        month=payload.month,
        year=payload.year
    )
    if result["status"] == "error":
        raise HTTPException(status_code=500, detail=result["message"])
    return result