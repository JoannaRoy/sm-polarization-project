from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import PlainTextResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from sqlalchemy import text

from db import connect
from db.views import TopicDetail, TopicSummary, topic_detail, topic_summary
from db.models import Topic
from main import (
    DEFAULT_TOPIC_CANDIDATES,
    configure_logging,
    match_post_topics,
    topic_response,
)

configure_logging()

app = FastAPI()


class TopicCandidate(BaseModel):
    id: str
    label: str
    polarity_target: str
    score: float


@app.get("/health")
def health():
    try:
        conn = connect()
        conn.execute(text("SELECT 1"))
        conn.close()
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"db unreachable: {exc}")
    return {"status": "ok", "db": "ok"}


@app.get("/topics", response_model=list[TopicSummary])
def topics():
    conn = connect()
    rows = conn.query(Topic).order_by(Topic.id).all()
    payload = [topic_summary(row) for row in rows]
    conn.close()
    return payload


@app.get("/topics/{topic_id}", response_model=TopicDetail)
def topic(topic_id: str):
    conn = connect()
    row = conn.get(Topic, topic_id)
    if row is None:
        conn.close()
        raise HTTPException(status_code=404, detail=f"Topic {topic_id} not found")
    payload = topic_detail(row)
    conn.close()
    return payload


@app.post("/match-post-topics", response_model=list[TopicCandidate])
def match_topics(post: dict, k: int = Query(DEFAULT_TOPIC_CANDIDATES, ge=1, le=10)):
    try:
        return match_post_topics(post, k=k)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@app.get("/topic-response/{topic_id}", response_class=PlainTextResponse)
def get_topic_response(topic_id: str):
    try:
        return topic_response(topic_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))


FRONTEND_DIST = Path(__file__).parent / "frontend" / "dist"
app.mount("/", StaticFiles(directory=FRONTEND_DIST, html=True), name="frontend")
