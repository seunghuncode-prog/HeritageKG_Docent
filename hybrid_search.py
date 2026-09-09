"""
Museum Hybrid Search
Vector DB (ChromaDB) 검색 결과를 Graph DB (Neo4j) 관계 정보로 보강하는 GraphRAG 모듈.

사용 예:
    from hybrid_search import HybridSearcher

    searcher = HybridSearcher(
        chroma_db_dir="C:/Users/user/Desktop/vector_db/chroma_museum_db",
        neo4j_uri="neo4j+ssc://...",
        neo4j_user="neo4j",
        neo4j_password="your_password",
    )

    results = searcher.search("고려시대 청자 유물", n_results=5)
    for r in results:
        print(r["context"])   # LLM에 넘길 텍스트 컨텍스트

    searcher.close()
"""

import os
import chromadb
from chromadb.utils import embedding_functions
from neo4j import GraphDatabase


CHROMA_DB_DIR     = os.environ.get("CHROMA_DB_DIR",     "vector_db/chroma_museum_db")
COLLECTION_NAME   = os.environ.get("COLLECTION_NAME",   "museum_relics")
EMBEDDING_MODEL   = os.environ.get("EMBEDDING_MODEL",   "jhgan/ko-sroberta-multitask")

NEO4J_URI         = os.environ.get("NEO4J_URI",         "bolt://localhost:7687")
NEO4J_USER        = os.environ.get("NEO4J_USER",        "neo4j")
NEO4J_PASSWORD    = os.environ.get("NEO4J_PASSWORD",    "")


class HybridSearcher:
    """
    GraphRAG 하이브리드 검색기.

    1. ChromaDB 벡터 검색 → 의미적으로 유사한 유물 Top-K
    2. Neo4j 관계 조회   → 해당 유물의 시대/재질/분류/작가/전시위치
    3. 통합 컨텍스트 반환 → LLM 프롬프트에 바로 사용 가능
    """

    def __init__(
        self,
        chroma_db_dir: str = CHROMA_DB_DIR,
        collection_name: str = COLLECTION_NAME,
        embedding_model: str = EMBEDDING_MODEL,
        neo4j_uri: str = NEO4J_URI,
        neo4j_user: str = NEO4J_USER,
        neo4j_password: str = NEO4J_PASSWORD,
    ):
        # ── ChromaDB 연결 ──────────────────────────────────────────────────────
        ef = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=embedding_model
        )
        chroma_client = chromadb.PersistentClient(path=chroma_db_dir)
        self._collection = chroma_client.get_collection(
            name=collection_name,
            embedding_function=ef,
        )

        # ── Neo4j 연결 ─────────────────────────────────────────────────────────
        self._neo4j = GraphDatabase.driver(
            neo4j_uri,
            auth=(neo4j_user, neo4j_password),
        )

    # ── 공개 API ───────────────────────────────────────────────────────────────

    def search(self, query: str, n_results: int = 5) -> list[dict]:
        """
        쿼리에 대해 벡터 검색 + 그래프 보강을 수행하고
        RAG에 사용할 수 있는 결과 리스트를 반환한다.

        반환 형식:
            [
                {
                    "rank": 1,
                    "distance": 0.22,
                    "chroma_id": "relic_329",
                    "relicId": 329,
                    "title": "흑갈유 병",
                    "다른명칭": "...",
                    "전시명칭": "...",
                    "소장품번호": "본관215",
                    "description": "...",
                    "graph": {
                        "국적": ["한국"],
                        "시대": ["조선"],
                        "재질": ["도자기", "흑유"],
                        "분류": ["공예", "도자기"],
                        "작가": [],
                        "전시위치": ["상설전시관 1층"],
                    },
                    "context": "...",  # LLM 입력용 통합 텍스트
                },
                ...
            ]
        """
        vector_hits = self._vector_search(query, n_results)
        enriched    = [self._enrich(rank, hit) for rank, hit in enumerate(vector_hits, 1)]
        return enriched

    def close(self):
        self._neo4j.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    # ── 내부 메서드 ────────────────────────────────────────────────────────────

    def _vector_search(self, query: str, n_results: int) -> list[dict]:
        """ChromaDB에서 상위 n_results개 유물을 반환한다."""
        raw = self._collection.query(
            query_texts=[query],
            n_results=n_results,
        )

        hits = []
        for i, chroma_id in enumerate(raw["ids"][0]):
            meta = raw["metadatas"][0][i]
            hits.append({
                "chroma_id": chroma_id,
                "distance":  raw["distances"][0][i],
                "relicId":   meta.get("relicId"),
                "title":     meta.get("title", ""),
                "다른명칭":  meta.get("다른명칭", ""),
                "전시명칭":  meta.get("전시명칭", ""),
                "소장품번호": meta.get("소장품번호", ""),
                "description": meta.get("description", ""),
            })
        return hits

    def _graph_enrich(self, 소장품번호: str) -> dict:
        """Neo4j에서 해당 유물의 관계 정보를 조회한다."""
        query = """
            MATCH (u:유물 {소장품번호: $번호})
            OPTIONAL MATCH (u)-[:국적]->(g:국적)
            OPTIONAL MATCH (u)-[:속한시대]->(s:시대)
            OPTIONAL MATCH (u)-[:재질로만들어짐]->(m:재질)
            OPTIONAL MATCH (u)-[:분류됨]->(c:분류)
            OPTIONAL MATCH (u)-[:만든작가]->(a:작가)
            OPTIONAL MATCH (u)-[:전시위치]->(e:전시위치)
            RETURN
                collect(DISTINCT g.name) AS 국적,
                collect(DISTINCT s.name) AS 시대,
                collect(DISTINCT m.name) AS 재질,
                collect(DISTINCT c.name) AS 분류,
                collect(DISTINCT a.name) AS 작가,
                collect(DISTINCT e.name) AS 전시위치
        """
        with self._neo4j.session() as session:
            record = session.run(query, 번호=소장품번호).single()

        if record is None:
            return {k: [] for k in ["국적", "시대", "재질", "분류", "작가", "전시위치"]}

        return {
            "국적":    list(record["국적"]),
            "시대":    list(record["시대"]),
            "재질":    list(record["재질"]),
            "분류":    list(record["분류"]),
            "작가":    list(record["작가"]),
            "전시위치": list(record["전시위치"]),
        }

    def _enrich(self, rank: int, hit: dict) -> dict:
        """벡터 검색 결과 1건에 그래프 정보를 붙이고 context 문자열을 생성한다."""
        graph = self._graph_enrich(hit["소장품번호"])

        context = _build_context(hit, graph)

        return {
            "rank":      rank,
            "distance":  hit["distance"],
            "chroma_id": hit["chroma_id"],
            "relicId":   hit["relicId"],
            "title":     hit["title"],
            "다른명칭":  hit["다른명칭"],
            "전시명칭":  hit["전시명칭"],
            "소장품번호": hit["소장품번호"],
            "description": hit["description"],
            "graph":     graph,
            "context":   context,
        }


# ── 헬퍼 ───────────────────────────────────────────────────────────────────────

def _build_context(hit: dict, graph: dict) -> str:
    """LLM 프롬프트에 삽입할 유물 정보 텍스트를 생성한다."""
    lines = [
        f"유물명: {hit['title']}",
    ]
    if hit["다른명칭"]:
        lines.append(f"다른명칭: {hit['다른명칭']}")
    if hit["전시명칭"]:
        lines.append(f"전시명칭: {hit['전시명칭']}")
    if hit["소장품번호"]:
        lines.append(f"소장품번호: {hit['소장품번호']}")

    for key in ["국적", "시대", "재질", "분류", "작가", "전시위치"]:
        vals = graph.get(key, [])
        if vals:
            lines.append(f"{key}: {', '.join(vals)}")

    if hit["description"]:
        lines.append(f"설명: {hit['description']}")

    return "\n".join(lines)


# ── CLI 간단 테스트 ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    if not NEO4J_PASSWORD:
        print("❌ NEO4J_PASSWORD 환경변수를 설정하세요.")
        print("   set NEO4J_PASSWORD=your_password")
        sys.exit(1)

    query = sys.argv[1] if len(sys.argv) > 1 else "고려시대 청자 유물"
    print(f"검색어: {query}\n")

    with HybridSearcher() as searcher:
        results = searcher.search(query, n_results=5)

    for r in results:
        print("=" * 80)
        print(f"[{r['rank']}위] 거리={r['distance']:.4f}  |  {r['chroma_id']}")
        print(r["context"])
