'''
최적화 방안
1. Query Caching (쿼리 캐싱)
API 서버 최전방에서 똑같은 질문을 기억
동일한 질문을 한 번 더 하면 즉시 답변을 반환

2. Hierarchical Judgment (계층적 문서 평가)
grade_documents 노드에서 문서가 통과될 때 무작정 LLM을 부르지 않고 키워드 매칭 수행
엉뚱하게 검색된 문서는 즉시 탈락시키고 1차 평가 통과한 문서만 LLM으로 넘겨 평가받음 무의미한 API 호출을 줄일 수 있음!!
'''


import os
import time
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

# 초기화
grader_client = genai.Client()
GRADER_MODEL  = "gemini-2.5-flash"
llm_generator = LLMGenerator()

app = FastAPI(title="Optimized Self RAG API")

# 전역 쿼리 캐시 딕셔너리
QUERY_CACHE = {}


class ChatRequest(BaseModel):
    query: str
    chat_history: List[dict] = []

class ChatResponse(BaseModel):
    answer: str
    sources: List[dict] = []

class GraphState(TypedDict):
    question:     str
    generation:   str
    documents:    list   # [{"title": ..., "context": ...}]
    sources:      list   # UI 카드용 원본 결과
    chat_history: list

# grader
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

# 노드 함수
def retrieve(state):
    question = state["question"]
    docs, sources = [], []

    print(f"\n[검색 시작] 질문: {question}")
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
    
    # 1차 필터용 키워드 추출
    keywords = [w for w in question.split() if len(w) >= 2]
    print(f"\n[계층적 판단 시작] 추출된 키워드: {keywords}")

    for d in state["documents"]:
        context_text = d['context']
        
        # 1차 필터
        passed_1st_filter = False
        for kw in keywords:
            if kw in context_text:
                passed_1st_filter = True
                break
                
        # 키워드가 하나라도 있는데 본문에 안 걸리면 바로 탈락 
        if len(keywords) > 0 and not passed_1st_filter:
            print(f"[1차 필터 탈락] 키워드 매칭 실패 (LLM 생략): {d['title']}")
            continue
            
        print(f"[1차 필터 통과] LLM 으로 넘김: {d['title']}")
        
        # --- 2차 필터 (Gemini LLM 기반 고비용 필터) ---
        user_msg = f"검색된 문서:\n\n{d['context']}\n\n사용자 질문: {question}"
        score = _call_grader(DOC_GRADE_SYSTEM, user_msg)
        
        if score == "yes":
            print(f" [2차 LLM 합격]: {d['title']}")
            filtered.append(d)
        else:
            print(f"[2차 LLM 탈락]: {d['title']}")

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
    print("\n[답변 생성 중...]")
    context      = "\n\n".join([d["context"] for d in state["documents"]])
    chat_history = state.get("chat_history", [])
    answer = llm_generator.generate(state["question"], context, chat_history)
    return {"generation": answer}


def grade_generation(state):
    # 무한루프 방지를 위해 평가 노드 이후 무조건 END로 가도록 수정
    pass


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


# 엔드포인트 
@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    start_time = time.time()
    query = req.query
    print(f"\n{'='*60}\n[새 요청 수신] {query}\n{'='*60}")
    
    # 1. 캐시 확인 (Query Caching)
    if query in QUERY_CACHE:
        elapsed = time.time() - start_time
        print(f" 캐시에서 즉시 반환 (소요 시간: {elapsed:.4f}초)")
        return QUERY_CACHE[query]
        
    print(f" 캐시에 없음")

    inputs = {
        "question":     query,
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

    response_data = ChatResponse(answer=answer, sources=sources)
    
    # 2. 캐시 저장
    QUERY_CACHE[query] = response_data
    elapsed = time.time() - start_time
    print(f" 쿼리 캐시에 저장 완료")

    return response_data


@app.get("/health")
def health():
    return {"status": "ok", "mode": "self_rag_optimized"}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8003, reload=False)
