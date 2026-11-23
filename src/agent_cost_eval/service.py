"""HTTP service for the support agent. Run: uvicorn agent_cost_eval.service:app --host 0.0.0.0 --port 8000"""
from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from langfuse import get_client
from pydantic import BaseModel, field_validator

from agent_cost_eval.config import load_settings
from agent_cost_eval.support import run_support_agent

settings = load_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    yield
    get_client().flush()  # don't lose buffered traces on shutdown or restart


app = FastAPI(title="support-agent", lifespan=lifespan)


class AskRequest(BaseModel):
    question: str
    user_id: str
    session_id: str

    @field_validator("question", "user_id", "session_id")
    @classmethod
    def not_blank(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("must not be blank")
        return v


class AskResponse(BaseModel):
    answer: str
    trace_url: str
    input_tokens: int
    output_tokens: int


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest) -> AskResponse:
    result, trace_id = run_support_agent(settings, req.question, req.user_id, req.session_id)
    if result.failed:
        raise HTTPException(status_code=502, detail=result.error)
    return AskResponse(
        answer=result.answer,
        trace_url=f"{settings.langfuse_host}/trace/{trace_id}",
        input_tokens=sum(s.input_tokens for s in result.steps),
        output_tokens=sum(s.output_tokens for s in result.steps),
    )
