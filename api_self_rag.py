"""
Graph RAG + Self RAG API 서버 (self_rag_chatbot.py 로직)
실행: python api_self_rag.py
포트: 8001
"""

import os
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())

from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, TypedDict
import uvicorn

from google import genai
from google.genai import types
from langgraph.graph import StateGraph, END

from hybrid_search import HybridSearcher
from llm.llm_generator import LLMGenerator
from prompts.grader_prompts import (
    DOC_GRADE_SYSTEM,
    GROUNDED_SYSTEM,
    ANSWER_GRADE_SYSTEM,
    REWRITE_SYSTEM,
)

# ── 초기화 ────────────────────────────────────────────────────────────────────
grader_client = genai.Client()
GRADER_MODEL  = "gemini-2.5-flash"
llm_generator = LLMGenerator()

app = FastAPI(title="Graph RAG + Self RAG API")


# ── 요청/응답 모델 ────────────────────────────────────────────────────────────
class ChatRequest(BaseModel):
    query: str
    chat_history: List[dict] = []


class ChatResponse(BaseModel):
    answer: str
    sources: List[dict] = []


# ── LangGraph 상태 ────────────────────────────────────────────────────────────
class GraphState(TypedDict):
    question:     str
    generation:   str
    documents:    list   # [{"title": ..., "context": ...}]
    sources:      list   # UI 카드용 원본 결과
    chat_history: list


# ── Grader ────────────────────────────────────────────────────────────────────
def _call_grader(system_instruction: str, user_content: str) -> str:
    try:
        response = grader_client.models.generate_content(
            model=GRADER_MODEL,
            contents=user_content,
            config=types.GenerateContentConfig(
                system_instruction=system_instruction,
                temperature=0,
            )
        )
        answer = response.text.strip().lower()
        return "yes" if "yes" in answer else "no"
    except Exception as e:
        print(f"Grader 호출 실패: {e}")
        return "yes"


# ── 노드 함수 ─────────────────────────────────────────────────────────────────
def retrieve(state):
    question = state["question"]
    docs, sources = [], []

    with HybridSearcher() as searcher:
        results = searcher.search(question, n_results=3)
        for r in results:
            docs.append({"title": r["title"], "context": r["context"]})
            sources.append({
                "rank":      r.get("rank", 1),
                "title":     r["title"],
                "소장품번호": r.get("소장품번호", ""),
                "graph":     r.get("graph", {}),
            })

    return {"documents": docs, "sources": sources}


def grade_documents(state):
    question = state["question"]
    filtered = []

    for d in state["documents"]:
        user_msg = f"검색된 문서:\n\n{d['context']}\n\n사용자 질문: {question}"
        score = _call_grader(DOC_GRADE_SYSTEM, user_msg)
        if score == "yes":
            filtered.append(d)

    return {"documents": filtered}


def decide_to_generate(state):
    return "generate"


def transform_query(state):
    question = state["question"]
    user_msg = f"원본 질문:\n\n{question}\n\n개선된 질문을 작성해주세요."
    try:
        response = grader_client.models.generate_content(
            model=GRADER_MODEL,
            contents=user_msg,
            config=types.GenerateContentConfig(
                system_instruction=REWRITE_SYSTEM,
                temperature=0,
            )
        )
        better = response.text.strip()
    except Exception:
        better = question
    return {"question": better}


def generate(state):
    context      = "\n\n".join([d["context"] for d in state["documents"]])
    chat_history = state.get("chat_history", [])
    answer = llm_generator.generate(state["question"], context, chat_history)
    return {"generation": answer}


def grade_generation(state):
    docs_text  = "\n\n".join([d["context"] for d in state["documents"]])
    generation = state["generation"]
    question   = state["question"]

    grounded = _call_grader(GROUNDED_SYSTEM,
        f"제공된 정보:\n\n{docs_text}\n\nAI 생성 답변: {generation}")

    if grounded == "yes":
        useful = _call_grader(ANSWER_GRADE_SYSTEM,
            f"사용자 질문:\n\n{question}\n\nAI 생성 답변: {generation}")
        return "relevant" if useful == "yes" else "not relevant"
    return "hallucination"


# ── LangGraph 조립 ────────────────────────────────────────────────────────────
def build_rag_graph():
    workflow = StateGraph(GraphState)

    workflow.add_node("retrieve",        retrieve)
    workflow.add_node("grade_documents", grade_documents)
    workflow.add_node("generate",        generate)
    workflow.add_node("transform_query", transform_query)

    workflow.set_entry_point("retrieve")
    workflow.add_edge("retrieve", "grade_documents")
    workflow.add_conditional_edges("grade_documents", decide_to_generate, {
        "transform_query": "transform_query",
        "generate":        "generate",
    })
    workflow.add_edge("transform_query", "retrieve")
    workflow.add_edge("generate", END)

    return workflow.compile()


rag_graph = build_rag_graph()


# ── 엔드포인트 ────────────────────────────────────────────────────────────────
@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    inputs = {
        "question":     req.query,
        "chat_history": req.chat_history,
        "documents":    [],
        "sources":      [],
        "generation":   "",
    }

    final_state = {}
    for output in rag_graph.stream(inputs):
        for value in output.values():
            final_state.update(value)

    answer  = final_state.get("generation", "답변을 생성하지 못했습니다.")
    sources = final_state.get("sources", [])

    return ChatResponse(answer=answer, sources=sources)


@app.get("/health")
def health():
    return {"status": "ok", "mode": "self_rag"}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001, reload=False)
