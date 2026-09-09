import os
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())                      

import json
from typing import List
from typing_extensions import TypedDict

from google import genai
from google.genai import types

from langgraph.graph import StateGraph, END

from hybrid_search import HybridSearcher          # 검색 
from llm.llm_generator import LLMGenerator         # 답변 생성

# Grader 프롬프트
from prompts.grader_prompts import (
    DOC_GRADE_SYSTEM,
    GROUNDED_SYSTEM,
    ANSWER_GRADE_SYSTEM,
    REWRITE_SYSTEM,
)



grader_client = genai.Client()
GRADER_MODEL = "gemini-2.5-flash"      
llm_generator = LLMGenerator()         


#grader call
def _call_grader(system_instruction: str, user_content: str) -> str:
    """Gemini에 yes/no 판정을 요청하고 결과를 소문자로 반환한다."""
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
        # "yes" 또는 "no"만 추출
        if "yes" in answer:
            return "yes"
        return "no"
    except Exception as e:
        print(f"Grader 호출 실패: {e}")
        return "yes" 


#langgraph 
class GraphState(TypedDict):
    question: str
    generation: str
    documents: list        # [{"title": ..., "context": ...}, ...]
    chat_history: list     # [{"role": "user", "content": ...}, ...]



#node 함수
def retrieve(state):
    """hybrid_search.py의 HybridSearcher로 검색"""
    print("==== [1. RETRIEVE (Hybrid GraphRAG 검색)] ====")
    question = state["question"]

    docs = []
    with HybridSearcher() as searcher:
        results = searcher.search(question, n_results=3)
        for r in results:
            docs.append({
                "title": r["title"],
                "context": r["context"],
            })
            print(f"검색됨: {r['title']}")

    return {"documents": docs}


def grade_documents(state):
    """검색된 문서가 질문과 관련 있는지 Gemini로 평가"""
    print("==== [2. GRADE DOCUMENTS (관련성 평가)] ====")
    question = state["question"]
    filtered = []

    for d in state["documents"]:
        user_msg = f"검색된 문서:\n\n{d['context']}\n\n사용자 질문: {question}"
        score = _call_grader(DOC_GRADE_SYSTEM, user_msg)

        if score == "yes":
            print(f"관련 있음: {d['title']}")
            filtered.append(d)
        else:
            print(f"관련 없음: {d['title']}")

    return {"documents": filtered}


def decide_to_generate(state):
    """관련 문서가 있으면 generate, 없어도 generate """
    print("==== [3. DECIDE (분기 결정)] ====")
    if not state["documents"]:
        print("  → 관련 문서 없음 → 그래도 답변 생성")
    else:
        print(f"  → 관련 문서 {len(state['documents'])}건 → 답변 생성")
    return "generate"


def transform_query(state):
    """질문을 검색에 최적화된 형태로 재작성"""
    print("==== [TRANSFORM QUERY (질문 재작성)] ====")
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
    except Exception as e:
        print(f"질문 재작성 실패: {e}")
        better = question

    print(f" → 재작성: {better}")
    return {"question": better}


def generate(state):
    """llm_generator.py (프롬프트 레이어링 내장)로 답변 생성"""
    print("==== [4. GENERATE (LLMGenerator로 답변 생성)] ====")

    # 문서 컨텍스트 합치기
    context = "\n\n".join([d["context"] for d in state["documents"]])
    chat_history = state.get("chat_history", [])

    # llm_generator.py 호출
    answer = llm_generator.generate(
        state["question"],
        context,
        chat_history,
    )

    return {"generation": answer}


def grade_generation(state):
    """환각 체크 + 질문 해결 여부 체크"""
    print("==== [5. CHECK HALLUCINATIONS (환각 및 정답 평가)] ====")
    documents_text = "\n\n".join([d["context"] for d in state["documents"]])
    generation = state["generation"]
    question = state["question"]

    # (1) 환각 평가
    grounded_msg = f"제공된 정보:\n\n{documents_text}\n\nAI 생성 답변: {generation}"
    grounded = _call_grader(GROUNDED_SYSTEM, grounded_msg)

    if grounded == "yes":
        print("사실 기반 답변")

        # (2) 답변 해결 여부 평가
        answer_msg = f"사용자 질문:\n\n{question}\n\nAI 생성 답변: {generation}"
        useful = _call_grader(ANSWER_GRADE_SYSTEM, answer_msg)

        if useful == "yes":
            print("질문에 정확히 답변함 → 완료")
            return "relevant"
        else:
            print("정확한 답변 실패→ 질문 재작성")
            return "not relevant"
    else:
        print("환각 감지 → 답변 재생성")
        return "hallucination"


#langgraph 조립
def build_graph():
    workflow = StateGraph(GraphState)

    workflow.add_node("retrieve", retrieve)
    workflow.add_node("grade_documents", grade_documents)
    workflow.add_node("generate", generate)
    workflow.add_node("transform_query", transform_query)

    workflow.set_entry_point("retrieve")
    workflow.add_edge("retrieve", "grade_documents")

    workflow.add_conditional_edges("grade_documents", decide_to_generate, {
        "transform_query": "transform_query",
        "generate": "generate",
    })

    workflow.add_edge("transform_query", "retrieve")

    workflow.add_conditional_edges("generate", grade_generation, {
        "hallucination": "generate",
        "not relevant": "transform_query",
        "relevant": END,
    })

    return workflow.compile()


# CLI 
if __name__ == "__main__":
    app = build_graph()
    chat_history = []

    print("=" * 60)
    print(" Hybrid GraphRAG + Self-RAG 문화유산 챗봇")
    print(" (종료: 'quit' 또는 '종료')")
    print("=" * 60)

    while True:
        user_question = input("\n질문: ")

        if user_question.strip().lower() in ["종료", "quit", "exit", "q"]:
            print("챗봇을 종료합니다.")
            break

        if not user_question.strip():
            continue

        chat_history.append({"role": "user", "content": user_question})

        inputs = {
            "question": user_question,
            "chat_history": chat_history,
        }

        # LangGraph 실행
        final_state = None
        for output in app.stream(inputs):
            for key, value in output.items():
                final_state = value

        answer = final_state.get("generation", "답변을 생성하지 못했습니다.")

        print(f"\n도슨트: {answer}")
        print("-" * 60)

        chat_history.append({"role": "assistant", "content": answer})
