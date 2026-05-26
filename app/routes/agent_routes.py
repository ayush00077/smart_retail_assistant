from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import sys
import os

# Make sure orchestrator is importable
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from orchestrator import route_query

router = APIRouter()

# -----------------------------
# Request / Response models
# -----------------------------
class QueryRequest(BaseModel):
    query: str

class QueryResponse(BaseModel):
    query:  str
    intent: str
    answer: str

# -----------------------------
# POST /api/agent/query
# -----------------------------
@router.post("/agent/query", response_model=QueryResponse)
async def agent_query(payload: QueryRequest):
    if not payload.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    try:
        result = route_query(payload.query)
        return QueryResponse(
            query=result["query"],
            intent=result["intent"],
            answer=result["answer"]
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# -----------------------------
# GET /api/agent/health
# -----------------------------
@router.get("/agent/health")
async def agent_health():
    return {
        "status": "ok",
        "agents": ["customer_qa", "data_analyst", "document_search"],
        "orchestrator": "active"
    }