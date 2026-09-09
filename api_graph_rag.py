"""
Graph RAG API 서버 (app.py 로직)
실행: python api_graph_rag.py
포트: 8000
"""

import os
import re
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())

from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import uvicorn

from graph.neo4j_client import Neo4jClient
from graph.graph_retriever import GraphRetriever
from vector_db.vector_retriever import VectorRetriever
from llm.llm_generator import LLMGenerator
from router.query_router import QueryRouter

# ── 객체 초기화 ───────────────────────────────────────────────────────────────
neo4j_client    = Neo4jClient()
graph_retriever = GraphRetriever(neo4j_client)
vector_retriever = VectorRetriever()
llm             = LLMGenerator()
router          = QueryRouter()

app = FastAPI(title="Graph RAG API")


# ── 요청/응답 모델 ────────────────────────────────────────────────────────────
class ChatRequest(BaseModel):
    query: str
    chat_history: List[dict] = []


class ChatResponse(BaseModel):
    answer: str
    sources: List[dict] = []


# ── 엔드포인트 ────────────────────────────────────────────────────────────────
@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    intent = router.route(req.query)

    sources = []

    # ── DESCRIPTION ──────────────────────────────────────────────────────────
    if intent == "description":
        artifact_name = (
            req.query
            .replace("알려줘", "")
            .replace("설명해줘", "")
            .strip()
        )

        # Graph 검색
        graph_result  = graph_retriever.retrieve_subgraph(artifact_name)
        graph_context = ""

        if graph_result:
            artifact = graph_result[0]["u"]
            graph_context += f"유물명: {artifact.get('title', '')}\n"
            graph_context += f"소장품번호: {artifact.get('소장품번호', '')}\n"
            if "전시명칭" in artifact:
                graph_context += f"전시명칭: {artifact.get('전시명칭', '')}\n"
            graph_context += "\n"
            for item in graph_result:
                relation  = item["r"][1]
                node_name = item["n"]["name"]
                graph_context += f"{relation}: {node_name}\n"

        # Vector 검색
        vector_result  = vector_retriever.search(artifact_name)
        vector_context = ""
        metas = vector_result["metadatas"][0]

        for meta in metas:
            vector_context += f"설명문: {meta['description']}\n\n"
            sources.append({
                "title":     meta.get("title", ""),
                "소장품번호": meta.get("소장품번호", ""),
                "description": meta.get("description", ""),
                "graph": {},
            })

        final_context = (
            f"[Graph 정보]\n{graph_context}\n"
            f"[설명문 정보]\n{vector_context}"
        )

    # ── RECOMMENDATION ───────────────────────────────────────────────────────
    elif intent == "recommendation":
        # 현재 유물명을 query에서 추출 (간단히 query 그대로 사용)
        similar_result = graph_retriever.retrieve_similar_artifacts(req.query)
        ctx = f"현재 유물: {req.query}\n\n비슷한 유물:\n"
        for item in similar_result:
            other = item["other"]
            ctx += f"- {other['title']}\n"
            sources.append({"title": other["title"], "소장품번호": "", "graph": {}})
        final_context = ctx

    else:
        final_context = ""

    answer = llm.generate(req.query, final_context, req.chat_history)
    return ChatResponse(answer=answer, sources=sources)


@app.get("/health")
def health():
    return {"status": "ok", "mode": "graph_rag"}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8002, reload=False)
