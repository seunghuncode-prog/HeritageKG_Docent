"""
국립중앙박물관 AI 도슨트 - Graph RAG API 서버
실행: uvicorn graph_rag_api:app --host 0.0.0.0 --port 8000 --reload
"""

import os
from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

from graph.neo4j_client import Neo4jClient
from graph.graph_retriever import GraphRetriever
from vector_db.vector_retriever import VectorRetriever
from llm.llm_generator import LLMGenerator
from router.query_router import QueryRouter
from prompts.style_prompt import Persona

# ── 초기화 (서버 시작 시 1회) ──────────────────────────────────────────────────
neo4j_client     = Neo4jClient()
graph_retriever  = GraphRetriever(neo4j_client)
vector_retriever = VectorRetriever()
llm              = LLMGenerator()
router           = QueryRouter()

# 세션별 상태 (chat_history, 마지막 유물명)
sessions: dict = {}

def get_session(session_id: str) -> dict:
    if session_id not in sessions:
        sessions[session_id] = {"chat_history": [], "current_artifact": None}
    return sessions[session_id]


PERSONA_MAP = {
    "어린이":      Persona.CHILD,
    "초보":        Persona.BEGINNER,
    "역사 전문가가": Persona.EXPERT,
}

# ── FastAPI ────────────────────────────────────────────────────────────────────
app = FastAPI(title="Graph RAG API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    question:   str
    persona:    str = "초보"       # "어린이" | "초보" | "역사 전문가가"
    session_id: str = "default"


class ChatResponse(BaseModel):
    answer: str
    intent: str = ""


# ── 엔드포인트 ─────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok", "server": "graph-rag-api"}


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    session = get_session(req.session_id)

    # intent 라우팅
    intent = router.route(req.question)

    # ── 컨텍스트 구성 ──────────────────────────────────────────────────────────
    if intent == "description":
        artifact_name = (
            req.question
            .replace("알려줘", "")
            .replace("설명해줘", "")
            .strip()
        )
        session["current_artifact"] = artifact_name

        # Graph 검색
        graph_result  = graph_retriever.retrieve_subgraph(artifact_name)
        graph_context = ""
        if graph_result:
            artifact = graph_result[0]["u"]
            graph_context += f"유물명: {artifact.get('title', '')}\n"
            graph_context += f"소장품번호: {artifact.get('소장품번호', '')}\n"
            if artifact.get("전시명칭"):
                graph_context += f"전시명칭: {artifact['전시명칭']}\n"
            if artifact.get("다른명칭"):
                graph_context += f"다른명칭: {artifact['다른명칭']}\n"
            graph_context += "\n"
            for item in graph_result:
                relation  = item["r"][1]
                node_name = item["n"].get("name", "")
                if node_name:
                    graph_context += f"{relation}: {node_name}\n"

        # Vector 검색
        vector_result  = vector_retriever.search(artifact_name)
        vector_context = ""
        for meta in vector_result.get("metadatas", [[]])[0]:
            desc = meta.get("description", "")
            if desc:
                vector_context += f"설명문: {desc}\n\n"

        final_context = f"[Graph 정보]\n{graph_context}\n[설명문 정보]\n{vector_context}"

    elif intent == "recommendation":
        current = session.get("current_artifact") or ""
        similar_result = graph_retriever.retrieve_similar_artifacts(current)
        context_lines  = [f"현재 유물: {current}\n\n비슷한 유물:"]
        for item in similar_result:
            other = item.get("other", {})
            title = other.get("title", "")
            if title:
                context_lines.append(f"- {title}")
        final_context = "\n".join(context_lines)

    else:
        final_context = ""

    # ── LLM 답변 생성 ──────────────────────────────────────────────────────────
    session["chat_history"].append({"role": "user", "content": req.question})

    answer = llm.generate(
        req.question,
        final_context,
        session["chat_history"],
    )

    session["chat_history"].append({"role": "assistant", "content": answer})

    return ChatResponse(answer=answer, intent=intent)


if __name__ == "__main__":
    uvicorn.run("graph_rag_api:app", host="0.0.0.0", port=8000, reload=True)
