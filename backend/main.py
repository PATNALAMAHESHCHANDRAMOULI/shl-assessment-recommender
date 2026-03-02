"""
main.py — FastAPI application for the SHL Assessment Recommender.
Endpoints: GET /health, POST /recommend
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from recommender import recommend

app = FastAPI(
    title="SHL Assessment Recommender API",
    description="AI-powered SHL assessment recommendation system",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    """Health check endpoint."""
    return {"status": "healthy", "service": "SHL Assessment Recommender"}


class QueryRequest(BaseModel):
    query: str


class Assessment(BaseModel):
    url: str
    name: str
    adaptive_support: str
    description: str
    duration: int
    remote_support: str
    test_type: list[str]


class RecommendResponse(BaseModel):
    recommended_assessments: list[Assessment]


@app.post("/recommend", response_model=RecommendResponse)
def recommend_assessments(req: QueryRequest):
    """
    Given a hiring query or job description, return 5–10 relevant SHL assessments.
    """
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    try:
        results = recommend(req.query, top_k=10)
        return {"recommended_assessments": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
