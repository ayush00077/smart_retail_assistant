from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from app.services.text_analytics_service import analyze_query

router = APIRouter()

class TextRequest(BaseModel):
    text: str

@router.post("/analytics/analyze")
async def analyze(payload: TextRequest):
    if not payload.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty")
    try:
        return analyze_query(payload.text)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))