from fastapi import FastAPI
from app.routes import upload
from app.routes.agent_routes import router as agent_router
from app.routes.ml_routes import router as ml_router
from app.routes.analytics_routes import router as analytics_router

app = FastAPI()

app.include_router(upload.router)
app.include_router(agent_router, prefix="/api")
app.include_router(ml_router, prefix="/api")
app.include_router(analytics_router, prefix="/api")

@app.get("/")
def home():
    return {"message": "smart retail assistant API"}




