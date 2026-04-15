from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import requests
try:
    from fastapi import FastAPI, HTTPException
    from pydantic import BaseModel, Field
except ModuleNotFoundError as exc:  # pragma: no cover - dependency path
    raise RuntimeError(
        "API dependencies are missing. Install project dependencies before using the API or query commands."
    ) from exc

from .config import load_app_config
from .prompts import build_low_confidence_response, build_query_prompt
from .retrieval import retrieve_chunks


class RetrieveRequest(BaseModel):
    question: str = Field(min_length=3)
    top_k: int | None = None


class QueryRequest(RetrieveRequest):
    model: str | None = None


@lru_cache(maxsize=2)
def get_config(root: str | None = None):
    return load_app_config(root or ".")


def query_ollama(host: str, model: str, system_prompt: str, user_prompt: str) -> str:
    response = requests.post(
        f"{host.rstrip('/')}/api/chat",
        json={
            "model": model,
            "stream": False,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        },
        timeout=180,
    )
    response.raise_for_status()
    payload = response.json()
    message = payload.get("message", {})
    content = message.get("content")
    if not content:
        raise RuntimeError(f"Unexpected Ollama response: {payload}")
    return content


def run_query_pipeline(question: str, root: str | Path = ".", model_name: str | None = None, top_k: int | None = None) -> dict[str, Any]:
    config = get_config(str(root))
    retrieval = retrieve_chunks(question, config, top_k=top_k)
    if not retrieval["grounded"]:
        payload = build_low_confidence_response(config.project["assistant"]["low_confidence_message"])
        payload["retrieved"] = retrieval["chunks"]
        payload["best_dense_score"] = retrieval["best_dense_score"]
        payload["best_bm25_score"] = retrieval["best_bm25_score"]
        return payload

    prompt = build_query_prompt(
        question=question,
        chunks=retrieval["chunks"],
        low_confidence_message=config.project["assistant"]["low_confidence_message"],
    )
    answer = query_ollama(
        host=config.project["models"]["ollama_host"],
        model=model_name or config.project["models"]["ollama_model_name"],
        system_prompt=config.project["assistant"]["system_prompt"],
        user_prompt=prompt,
    )
    citations = [chunk["citation_label"] for chunk in retrieval["chunks"]]
    confidence = "high" if retrieval["best_dense_score"] >= 0.45 else "medium"
    return {
        "answer": answer,
        "citations": citations,
        "confidence": confidence,
        "grounded": True,
        "best_dense_score": retrieval["best_dense_score"],
        "best_bm25_score": retrieval["best_bm25_score"],
        "retrieved": retrieval["chunks"],
    }


def create_app(root: str | Path = ".") -> FastAPI:
    app = FastAPI(title="Bangladesh Law Ollama API", version="0.1.0")
    get_config(str(root))

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/retrieve")
    def retrieve(request: RetrieveRequest) -> dict[str, Any]:
        try:
            config = get_config(str(root))
            return retrieve_chunks(request.question, config, top_k=request.top_k)
        except Exception as exc:  # pragma: no cover - FastAPI passthrough
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    @app.post("/query")
    def query(request: QueryRequest) -> dict[str, Any]:
        try:
            return run_query_pipeline(request.question, root=root, model_name=request.model, top_k=request.top_k)
        except requests.HTTPError as exc:  # pragma: no cover - runtime network path
            raise HTTPException(status_code=502, detail=str(exc)) from exc
        except Exception as exc:  # pragma: no cover - FastAPI passthrough
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    return app
