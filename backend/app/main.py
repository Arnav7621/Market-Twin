
from fastapi import FastAPI
from app.queue_mind.api.fill_router import router as fill_router
app = FastAPI(
    title="MarketTwin API",
    description="AI-powered financial market digital twin",
    version="1.0.0"
)
app.include_router(fill_router)
@app.get("/")
def root():
    return {
        "project": "MarketTwin",
        "status": "running"
    }

@app.get("/health")
def health_check():
    return {"status": "healthy"}
