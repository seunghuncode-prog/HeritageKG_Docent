from graph.neo4j_client import Neo4jClient
from graph.graph_retriever import GraphRetriever

from vector_db.vector_retriever import VectorRetriever

from llm.llm_generator import LLMGenerator

from router.query_router import QueryRouter


# ==================================================
# 객체 생성
# ==================================================

neo4j_client = Neo4jClient()

graph_retriever = GraphRetriever(
    neo4j_client
)

vector_retriever = VectorRetriever()

llm = LLMGenerator()

router = QueryRouter()

chat_history = []

current_artifact = None


# ==================================================
# 채팅 루프
# ==================================================

while True:

    query = input("질문: ")

    if query == "exit":
        break

    # ==================================================
    # history 저장
    # ==================================================

    chat_history.append({
        "role": "user",
        "content": query
    })

    # ==================================================
    # intent routing
    # ==================================================

    intent = router.route(query)

    # ==================================================
    # DESCRIPTION
    # ==================================================

    if intent == "description":

        artifact_name = (
            query
            .replace("알려줘", "")
            .replace("설명해줘", "")
            .strip()
        )

        current_artifact = artifact_name

        # ---------------- graph retrieval ----------------

        graph_result = (
            graph_retriever.retrieve_subgraph(
                artifact_name
            )
        )

        graph_context = ""

        if graph_result:

            artifact = graph_result[0]["u"]

            graph_context += (
                f"유물명: "
                f"{artifact.get('title', '')}\n"
            )

            graph_context += (
                f"소장품번호: "
                f"{artifact.get('소장품번호', '')}\n"
            )

            if "전시명칭" in artifact:

                graph_context += (
                    f"전시명칭: "
                    f"{artifact.get('전시명칭', '')}\n"
                )

            if "다른명칭" in artifact:

                graph_context += (
                    f"다른명칭: "
                    f"{artifact.get('다른명칭', '')}\n"
                )

            graph_context += "\n"

            for item in graph_result:

                relation = item["r"][1]
                node_name = item["n"]["name"]

                graph_context += (
                    f"{relation}: "
                    f"{node_name}\n"
                )

        # ---------------- vector retrieval ----------------

        vector_result = (
            vector_retriever.search(
                artifact_name
            )
        )

        vector_context = ""

        metas = vector_result["metadatas"][0]

        for meta in metas:

            vector_context += (
                f"설명문: "
                f"{meta['description']}\n\n"
            )

        final_context = f"""
[Graph 정보]
{graph_context}

[설명문 정보]
{vector_context}
"""

    # ==================================================
    # RECOMMENDATION
    # ==================================================

    elif intent == "recommendation":

        similar_result = (
            graph_retriever
            .retrieve_similar_artifacts(
                current_artifact
            )
        )

        recommendation_context = (
            f"현재 유물: "
            f"{current_artifact}\n\n"
        )

        recommendation_context += (
            "비슷한 유물:\n"
        )

        for item in similar_result:

            other = item["other"]

            recommendation_context += (
                f"- {other['title']}\n"
            )

        final_context = recommendation_context

    # ==================================================
    # context 출력
    # ==================================================

    print(final_context)

    # ==================================================
    # LLM
    # ==================================================

    answer = llm.generate(
        query,
        final_context,
        chat_history
    )

    print("\n===== 최종 답변 =====\n")

    print(answer)

    # ==================================================
    # 답변 저장
    # ==================================================

    chat_history.append({
        "role": "assistant",
        "content": answer
    })