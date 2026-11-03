
from fastapi import FastAPI

app = FastAPI(
    title="MarketTwin API",
    description="AI-powered financial market digital twin",
    version="1.0.0"
)

@app.get("/")
def root():
    return {
        "project": "MarketTwin",
        "status": "running"
    }

@app.get("/health")
def health_check():
    return {"status": "healthy"}
