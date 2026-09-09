# ragas가 내부적으로 langchain_community.chat_models.vertexai를 import하는데
# 최신 langchain_community에서 제거됨 → dummy stub으로 우회
import sys, types
_stub = types.ModuleType("langchain_community.chat_models.vertexai")
_stub.ChatVertexAI = type("ChatVertexAI", (), {})
sys.modules.setdefault("langchain_community.chat_models.vertexai", _stub)

import os
import json
import sys
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())

SAMPLES_CACHE = "ragas_samples.json"

from ragas import evaluate, EvaluationDataset
from ragas.metrics import LLMContextRecall, Faithfulness, FactualCorrectness
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.run_config import RunConfig
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from api_self_rag_optimized import build_rag_graph


# ── OpenAI Evaluator 초기화 ───────────────────────────────────────────────────
evaluator_llm = LangchainLLMWrapper(ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0,
    api_key=os.getenv("OPENAI_API_KEY"),
    timeout=120,
))

evaluator_embeddings = LangchainEmbeddingsWrapper(OpenAIEmbeddings(
    model="text-embedding-3-small",
    api_key=os.getenv("OPENAI_API_KEY"),
))


# ── 테스트셋 ──────────────────────────────────────────────────────────────────
# (question, ground_truth, error_type)
TEST_CASES = [

    # ── Type 1: 시대 오류 ────────────────────────────────────────────────────
    (
        "고려시대에 그려진 이암의 어미개와 강아지에 대해 알려줘",
        "어미개와 강아지는 고려시대가 아닌 조선시대 화가 이암(1499-?)의 작품입니다.",
        "시대오류",
    ),
    (
        "통일신라 시대에 만들어진 청자 상감 구름 학 무늬 항아리를 설명해줘",
        "청자 상감 구름 학 무늬 항아리는 통일신라가 아닌 고려시대 유물입니다.",
        "시대오류",
    ),
    (
        "조선시대에 제작된 감산사 미륵보살입상의 특징이 뭐야?",
        "감산사 미륵보살입상은 조선이 아닌 통일신라 719년에 김지성이 조성한 국보입니다.",
        "시대오류",
    ),
    (
        "고려시대 도예로 만들어진 백자 철화 포도 원숭이무늬 항아리에 대해 알려줘",
        "백자 철화 포도 원숭이무늬 항아리는 고려가 아닌 조선시대 국보입니다.",
        "시대오류",
    ),
    (
        "조선 초기 무장 이성계가 황산에서 왜구를 물리쳤다는 황산대첩비명에 대해 알려줘",
        "황산대첩은 1380년 고려 말의 사건입니다. 이성계는 당시 고려의 무장이었으며 조선 건국은 1392년입니다.",
        "시대오류",
    ),

    # ── Type 2: 인물 오류 ────────────────────────────────────────────────────
    (
        "김수철이 그린 어미개와 강아지의 화풍에 대해 설명해줘",
        "어미개와 강아지의 작가는 김수철이 아닌 이암(1499-?)입니다. 김수철은 겨울 산수의 작가입니다.",
        "인물오류",
    ),
    (
        "이암이 그린 겨울 산수에 담긴 시문 내용을 알려줘",
        "겨울 산수의 작가는 이암이 아닌 김수철(金秀哲)입니다.",
        "인물오류",
    ),
    (
        "진흥왕이 돌아가신 어머니를 기리며 만든 감산사 미륵보살입상을 설명해줘",
        "감산사 미륵보살입상은 진흥왕이 아닌 김지성(金志誠)이 719년 어머니를 위해 조성했습니다.",
        "인물오류",
    ),
    (
        "강감찬이 황산에서 왜구를 크게 물리쳤다는 승전비에 대해 알려줘",
        "황산대첩은 강감찬이 아닌 이성계의 승리입니다. 강감찬은 귀주대첩으로 유명합니다.",
        "인물오류",
    ),
    (
        "이성계가 흥국사에 탑을 세웠다는 천희오년 탑명에 대해 설명해줘",
        "흥국사 탑은 이성계가 아닌 강감찬이 세운 것입니다. 천희 5년은 1021년 고려 현종 12년입니다.",
        "인물오류",
    ),

    # ── Type 3: 출처 위조 ────────────────────────────────────────────────────
    (
        "조선왕조실록에 기록된 청자 상감 구름 학 무늬 항아리의 역사를 알려줘",
        "청자 상감 구름 학 무늬 항아리는 고려 유물로, 조선왕조실록에는 관련 기록이 없습니다.",
        "출처위조",
    ),
    (
        "삼국사기에 기록된 금동약사불입상의 제작 경위를 알려줘",
        "금동약사불입상의 제작 경위가 삼국사기에 기록되었다는 근거는 없습니다.",
        "출처위조",
    ),
    (
        "고려도경에 나오는 이암의 동물 그림 관련 내용을 알려줘",
        "고려도경은 1124년 문서로 이암(1499-?) 출생보다 375년 앞서 기록 자체가 불가능합니다.",
        "출처위조",
    ),
    (
        "인종실록에 기록된 감산사 미륵보살입상 조성 배경을 설명해줘",
        "감산사 미륵보살입상의 조성 배경은 광배 뒷면 명문에 기록되어 있으며 인종실록과는 무관합니다.",
        "출처위조",
    ),
    (
        "삼국유사에 등장하는 고려 청동 공양탑에 대해 알려줘",
        "고려 청동 공양탑은 삼국유사에 등장하지 않습니다. 삼국유사는 삼국·통일신라 시대 기록입니다.",
        "출처위조",
    ),

    # ── Type 4: 인물연관 ─────────────────────────────────────────────────────
    (
        "이암과 같은 시대에 활동한 다른 조선 화가의 작품을 알려줘",
        "이암과 같은 조선시대 화가로 김수철(겨울 산수), 양팽손(산수 그림) 등이 있습니다.",
        "인물연관",
    ),
    (
        "김수철과 비슷한 화풍의 조선 말기 화가 작품을 추천해줘",
        "김수철은 조선 말기 이색적인 화풍을 구사했으며, 양팽손의 산수 그림도 조선시대 선비화가의 작품입니다.",
        "인물연관",
    ),
    (
        "강감찬과 관련된 고려시대 유물이나 기록물이 있나요?",
        "강감찬이 흥국사에 세운 탑의 명문을 탁본한 천희오년 탑명 탑본(본관290)이 있습니다.",
        "인물연관",
    ),

    # ── Type 5: 지역연관 ─────────────────────────────────────────────────────
    (
        "경주에서 옮겨온 불상을 알려줘",
        "경주 남산 삼릉계에서 1915년 서울로 옮겨진 석조약사불좌상(본관1957)이 있습니다.",
        "지역연관",
    ),
    (
        "경상북도 출토 통일신라 불상을 소개해줘",
        "경상북도 출토 통일신라 불상으로는 감산사 석조미륵보살입상과 감산사 석조아미타불입상(국보)이 있습니다.",
        "지역연관",
    ),
    (
        "개성 흥국사와 관련된 고려 유물이 있나요?",
        "고려 명장 강감찬이 경기도 개성 흥국사 터에 세운 석탑의 명문을 탁본한 천희오년 탑명 탑본이 있습니다.",
        "지역연관",
    ),

    # ── Type 6: 용도유사 ─────────────────────────────────────────────────────
    (
        "액체나 음식을 담는 용도로 쓰인 도자기 유물을 추천해줘",
        "흑갈유 병(본관215), 청자 음각 모란 상감 보자기무늬 뚜껑 매병(본관1981), 청자 상감 구름 학 무늬 항아리(본관1984) 등이 있습니다.",
        "용도유사",
    ),
    (
        "저장이나 운반 용도의 고려 청자 유물을 알려줘",
        "고려 청자 저장 운반 용기로는 청자 음각 모란 상감 보자기무늬 뚜껑 매병(보물)과 청자 상감 구름 학 무늬 항아리가 있습니다.",
        "용도유사",
    ),
    (
        "불교 의식에 사용된 금속 유물에는 어떤 것이 있나요?",
        "불교 의식 관련 금속 유물로는 고려시대 청동 공양탑(본관376)과 각종 금동약사불입상들이 있습니다.",
        "용도유사",
    ),

    # ── Type 7: 재료유사 ─────────────────────────────────────────────────────
    (
        "금동으로 만든 불상 유물을 모두 알려줘",
        "금동 불상으로는 금동약사불입상(본관244, 신라), 금동 약사불 입상(본관324, 325, 326)과 금동 석가불 입상(본관327) 등이 있습니다.",
        "재료유사",
    ),
    (
        "청자 재질로 만들어진 고려 유물을 추천해줘",
        "고려 청자 유물로는 청자 음각 모란 상감 보자기무늬 뚜껑 매병(보물), 청자 상감 구름 학 무늬 항아리, 청자 상감 연꽃 넝쿨 무늬 합 등이 있습니다.",
        "재료유사",
    ),
    (
        "종이 재질의 조선시대 회화 유물을 소개해줘",
        "종이 재질의 조선 회화로는 이암의 어미개와 강아지(본관255), 김수철의 겨울 산수(본관281), 양팽손의 산수 그림(본관2034) 등이 있습니다.",
        "재료유사",
    ),
]


# ── Self-RAG 파이프라인 ────────────────────────────────────────────────────────
rag_app = build_rag_graph()


def run_self_rag(question: str) -> tuple[str, list[str]]:
    """Self-RAG 파이프라인 실행 → (answer, contexts) 반환"""
    inputs = {
        "question":     question,
        "chat_history": [],
        "documents":    [],
        "sources":      [],
        "generation":   "",
    }
    final_state: dict = {}
    for output in rag_app.stream(inputs):
        for value in output.values():
            final_state.update(value)

    answer   = final_state.get("generation", "답변을 생성하지 못했습니다.")
    contexts = [d["context"] for d in final_state.get("documents", [])]
    return answer, contexts


# ── 평가 실행 ─────────────────────────────────────────────────────────────────
def main():
    eval_only = "--eval-only" in sys.argv

    print("=" * 60)
    print("  Museum RAG - RAGAS 평가 (Gemini 2.5 Flash)")
    print(f"  총 {len(TEST_CASES)}개 질문 / 7가지 유형")
    print("=" * 60)

    # ── RAG 파이프라인 실행 or 캐시 로드 ────────────────────────────────────
    if eval_only and os.path.exists(SAMPLES_CACHE):
        print(f"\n[캐시 로드] {SAMPLES_CACHE} 에서 불러옵니다 (RAG 파이프라인 생략)")
        with open(SAMPLES_CACHE, encoding="utf-8") as f:
            saved = json.load(f)
        samples = saved["samples"]
        meta    = saved["meta"]
    else:
        samples = []
        meta    = []

        for i, (question, ground_truth, error_type) in enumerate(TEST_CASES, 1):
            print(f"\n[{i:02d}/{len(TEST_CASES)}] ({error_type}) {question[:45]}...")
            try:
                answer, contexts = run_self_rag(question)
                samples.append({
                    "user_input":         question,
                    "retrieved_contexts": contexts if contexts else ["관련 문서 없음"],
                    "response":           answer,
                    "reference":          ground_truth,
                })
                meta.append({"question": question, "error_type": error_type})
                print(f"  → 완료 (context {len(contexts)}개)")
            except Exception as e:
                print(f"  [ERROR] {e}")

        # 중간 결과 저장 (다음에 --eval-only로 재사용 가능)
        with open(SAMPLES_CACHE, "w", encoding="utf-8") as f:
            json.dump({"samples": samples, "meta": meta}, f, ensure_ascii=False, indent=2)
        print(f"\n[저장] RAG 결과를 {SAMPLES_CACHE} 에 저장했습니다.")

    if not samples:
        print("평가할 샘플이 없습니다.")
        return

    dataset = EvaluationDataset.from_list(samples)

    print("\n" + "=" * 60)
    print("  RAGAS 평가 실행 중 (Gemini evaluator)...")
    print("=" * 60)

    result = evaluate(
        dataset=dataset,
        metrics=[
            LLMContextRecall(),
            Faithfulness(),
            FactualCorrectness(),
        ],
        llm=evaluator_llm,
        embeddings=evaluator_embeddings,
        run_config=RunConfig(timeout=120, max_retries=3),
    )

    print("\n" + "=" * 60)
    print("  전체 평가 결과")
    print("=" * 60)
    print(result)

    # 유형별 세부 분석
    df = result.to_pandas()
    df["error_type"] = [m["error_type"] for m in meta]

    print("\n[유형별 FactualCorrectness 평균]")
    print(df.groupby("error_type")["factual_correctness"].mean().round(3).to_string())

    output_path = "ragas_result.csv"
    df.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"\n결과 저장: {output_path}")


if __name__ == "__main__":
    main()
