"""
Museum Graph DB Builder  (강화 버전)
museum_collections.json → Neo4j 그래프 구축 스크립트

노드: 유물, 국적, 시대, 재질, 분류, 전시위치, 작가

관계:
    [기본]
    (유물) -[:국적]-----------> (국적)
    (유물) -[:속한시대]-------> (시대  {시작년도, 종료년도})
    (유물) -[:재질로만들어짐]-> (재질)
    (유물) -[:분류됨]---------->(분류)
    (유물) -[:전시위치]-------> (전시위치)
    (유물) -[:만든작가]-------> (작가)

    [계층/시계열]
    (분류) -[:상위분류]-------> (분류)          ← 분류 계층 탐색
    (시대) -[:다음시대]-------> (시대)          ← 시간 순서
    (국적) -[:시대포함]-------> (시대)          ← 국가별 시대 묶음

사용법:
    set NEO4J_URI=bolt://localhost:7687
    set NEO4J_USER=neo4j
    set NEO4J_PASSWORD=your_neo4j_password
    python neo4j_builder.py
"""

import json
import os
from dotenv import load_dotenv

load_dotenv()
from neo4j import GraphDatabase

# ── 환경변수 ─────────────────────────────────────────────────────────────────
NEO4J_URI      = os.environ.get("NEO4J_URI",      "bolt://localhost:7687")
NEO4J_USER     = os.environ.get("NEO4J_USER",     "neo4j")
NEO4J_PASSWORD = os.environ.get("NEO4J_PASSWORD", "")
DATA_FILE      = os.environ.get("DATA_FILE",      "museum_collections.json")
BATCH_SIZE     = 50


# ── 시대 메타데이터 (하드코딩) ────────────────────────────────────────────────

ERA_YEARS = {
    # 한국
    "구석기":     (-700000, -10000),
    "신석기":     (-8000,   -1500),
    "청동기":     (-1500,   -300),
    "고조선":     (-2333,   -108),
    "고조선·부여·삼한": (-2333, 300),
    "삼한":       (-100,    300),
    "고구려":     (-37,     668),
    "백제":       (-18,     660),
    "신라":       (-57,     935),
    "가야":       (42,      562),
    "통일신라":   (668,     935),
    "발해":       (698,     926),
    "고려":       (918,     1392),
    "조선":       (1392,    1897),
    "대한제국":   (1897,    1910),
    "일제강점기": (1910,    1945),
    "근현대":     (1900,    2026),
    # 중국
    "상":   (-1600, -1046),
    "주":   (-1046, -256),
    "진":   (-221,  -206),
    "한":   (-206,  220),
    "위진남북조": (220, 589),
    "수":   (581,   618),
    "당":   (618,   907),
    "오대": (907,   960),
    "송":   (960,   1279),
    "원":   (1271,  1368),
    "명":   (1368,  1644),
    "청":   (1644,  1912),
    # 일본
    "조몬":   (-14000, -300),
    "야요이": (-300,   250),
    "고훈":   (250,    538),
    "아스카": (538,    710),
    "나라":   (710,    794),
    "헤이안": (794,    1185),
    "가마쿠라": (1185, 1333),
    "무로마치": (1336, 1573),
    "모모야마": (1573, 1615),
    "에도":   (1615,  1868),
    "메이지": (1868,  1912),
    "다이쇼": (1912,  1926),
    "쇼와":   (1926,  1989),
    # 공통
    "근대":   (1800,  1945),
    "현대":   (1945,  2026),
    "선사":   (-700000, -57),
}

# 한국 시대 순서
ERA_SEQ_KOREA = [
    "구석기", "신석기", "청동기", "고조선", "삼한",
    "고구려", "백제", "신라", "가야",
    "통일신라", "발해", "고려", "조선", "대한제국", "일제강점기",
]
# 중국 시대 순서
ERA_SEQ_CHINA = [
    "상", "주", "진", "한", "위진남북조", "수", "당", "오대", "송", "원", "명", "청",
]
# 일본 시대 순서
ERA_SEQ_JAPAN = [
    "조몬", "야요이", "고훈", "아스카", "나라", "헤이안",
    "가마쿠라", "무로마치", "모모야마", "에도", "메이지", "다이쇼", "쇼와",
]

# 국적 → 시대 목록
NATIONALITY_ERAS = {
    "한국": ERA_SEQ_KOREA,
    "중국": ERA_SEQ_CHINA,
    "일본": ERA_SEQ_JAPAN,
}


# ── 파싱 함수 ────────────────────────────────────────────────────────────────

def parse_nationality_era(value: str):
    """'한국 - 신라' → ('한국', '신라')  /  '중국' → ('중국', None)"""
    if not value or not value.strip():
        return None, None
    parts = [p.strip() for p in value.split(" - ")]
    nationality = parts[0] or None
    era = parts[1] if len(parts) > 1 and parts[1] else None
    return nationality, era


def parse_multival(value: str, sep: str = " - ") -> list[str]:
    """'금속 - 금동' → ['금속', '금동']"""
    if not value or not value.strip():
        return []
    return [p.strip() for p in value.split(sep) if p.strip()]


def parse_location(value: str) -> list[str]:
    """'가네코실 / 기증3' → ['가네코실', '기증3']"""
    if not value or not value.strip():
        return []
    return [p.strip() for p in value.split(" / ") if p.strip()]


# ── DB 초기화 ─────────────────────────────────────────────────────────────────

def create_constraints(session):
    stmts = [
        "CREATE CONSTRAINT IF NOT EXISTS FOR (u:유물)     REQUIRE u.소장품번호 IS UNIQUE",
        "CREATE CONSTRAINT IF NOT EXISTS FOR (n:국적)     REQUIRE n.name IS UNIQUE",
        "CREATE CONSTRAINT IF NOT EXISTS FOR (s:시대)     REQUIRE s.name IS UNIQUE",
        "CREATE CONSTRAINT IF NOT EXISTS FOR (m:재질)     REQUIRE m.name IS UNIQUE",
        "CREATE CONSTRAINT IF NOT EXISTS FOR (c:분류)     REQUIRE c.name IS UNIQUE",
        "CREATE CONSTRAINT IF NOT EXISTS FOR (e:전시위치) REQUIRE e.name IS UNIQUE",
        "CREATE CONSTRAINT IF NOT EXISTS FOR (a:작가)     REQUIRE a.name IS UNIQUE",
    ]
    for stmt in stmts:
        try:
            session.run(stmt)
        except Exception:
            pass  # 이미 존재하면 무시


# ── 배치 삽입: 유물 + 기본 관계 ──────────────────────────────────────────────

def insert_batch(tx, batch: list[dict]):
    """UNWIND 배치 — 유물 노드 및 6가지 기본 관계"""

    # 1. 유물 노드
    tx.run("""
        UNWIND $rows AS row
        MERGE (u:유물 {소장품번호: row.소장품번호})
        SET u.relicId  = row.relicId,
            u.title    = row.title,
            u.다른명칭  = row.다른명칭,
            u.전시명칭  = row.전시명칭
    """, rows=[{
        "소장품번호": item.get("소장품번호", ""),
        "relicId":    item.get("relicId"),
        "title":      item.get("title", ""),
        "다른명칭":   item.get("다른명칭", ""),
        "전시명칭":   item.get("전시명칭", ""),
    } for item in batch])

    # 2. 국적
    nat_rows = []
    for item in batch:
        nat, _ = parse_nationality_era(item.get("국적/시대", ""))
        if nat:
            nat_rows.append({"소장품번호": item["소장품번호"], "name": nat})
    if nat_rows:
        tx.run("""
            UNWIND $rows AS row
            MERGE (n:국적 {name: row.name})
            WITH n, row
            MATCH (u:유물 {소장품번호: row.소장품번호})
            MERGE (u)-[:국적]->(n)
        """, rows=nat_rows)

    # 3. 시대
    era_rows = []
    for item in batch:
        _, era = parse_nationality_era(item.get("국적/시대", ""))
        if era:
            era_rows.append({"소장품번호": item["소장품번호"], "name": era})
    if era_rows:
        tx.run("""
            UNWIND $rows AS row
            MERGE (s:시대 {name: row.name})
            WITH s, row
            MATCH (u:유물 {소장품번호: row.소장품번호})
            MERGE (u)-[:속한시대]->(s)
        """, rows=era_rows)

    # 4. 재질 (멀티레벨)
    mat_rows = []
    for item in batch:
        for name in parse_multival(item.get("재질", "")):
            mat_rows.append({"소장품번호": item["소장품번호"], "name": name})
    if mat_rows:
        tx.run("""
            UNWIND $rows AS row
            MERGE (m:재질 {name: row.name})
            WITH m, row
            MATCH (u:유물 {소장품번호: row.소장품번호})
            MERGE (u)-[:재질로만들어짐]->(m)
        """, rows=mat_rows)

    # 5. 분류 (멀티레벨)
    cat_rows = []
    for item in batch:
        for name in parse_multival(item.get("분류", "")):
            cat_rows.append({"소장품번호": item["소장품번호"], "name": name})
    if cat_rows:
        tx.run("""
            UNWIND $rows AS row
            MERGE (c:분류 {name: row.name})
            WITH c, row
            MATCH (u:유물 {소장품번호: row.소장품번호})
            MERGE (u)-[:분류됨]->(c)
        """, rows=cat_rows)

    # 6. 전시위치 (' / ' 분리)
    loc_rows = []
    for item in batch:
        for name in parse_location(item.get("전시위치", "")):
            loc_rows.append({"소장품번호": item["소장품번호"], "name": name})
    if loc_rows:
        tx.run("""
            UNWIND $rows AS row
            MERGE (e:전시위치 {name: row.name})
            WITH e, row
            MATCH (u:유물 {소장품번호: row.소장품번호})
            MERGE (u)-[:전시위치]->(e)
        """, rows=loc_rows)

    # 7. 작가
    art_rows = []
    for item in batch:
        artist = item.get("작가", "").strip()
        if artist:
            art_rows.append({"소장품번호": item["소장품번호"], "name": artist})
    if art_rows:
        tx.run("""
            UNWIND $rows AS row
            MERGE (a:작가 {name: row.name})
            WITH a, row
            MATCH (u:유물 {소장품번호: row.소장품번호})
            MERGE (u)-[:만든작가]->(a)
        """, rows=art_rows)


# ── 강화 관계 구축 ───────────────────────────────────────────────────────────

def build_category_hierarchy(session, data: list[dict]):
    """(하위분류) -[:상위분류]-> (상위분류) 계층 생성"""
    rows, seen = [], set()
    for item in data:
        cats = parse_multival(item.get("분류", ""))
        for i in range(1, len(cats)):
            key = (cats[i], cats[i - 1])
            if key not in seen:
                rows.append({"child": cats[i], "parent": cats[i - 1]})
                seen.add(key)
    if rows:
        session.run("""
            UNWIND $rows AS row
            MERGE (child:분류  {name: row.child})
            MERGE (parent:분류 {name: row.parent})
            MERGE (child)-[:상위분류]->(parent)
        """, rows=rows)
    print(f"   분류 계층 관계: {len(rows)}개")


def build_era_metadata(session):
    """시대 연도 범위 설정 + 시대 순서 관계 + 국적-시대 연결"""

    # 시대 연도 속성
    era_rows = [{"name": k, "start": v[0], "end": v[1]} for k, v in ERA_YEARS.items()]
    session.run("""
        UNWIND $rows AS row
        MERGE (s:시대 {name: row.name})
        SET s.시작년도 = row.start,
            s.종료년도 = row.end
    """, rows=era_rows)
    print(f"   시대 연도 범위: {len(era_rows)}개 설정")

    # 시대 순서 관계 (다음시대)
    seq_rows = []
    for seq in [ERA_SEQ_KOREA, ERA_SEQ_CHINA, ERA_SEQ_JAPAN]:
        for i in range(len(seq) - 1):
            seq_rows.append({"a": seq[i], "b": seq[i + 1]})
    session.run("""
        UNWIND $rows AS row
        MERGE (a:시대 {name: row.a})
        MERGE (b:시대 {name: row.b})
        MERGE (a)-[:다음시대]->(b)
    """, rows=seq_rows)
    print(f"   시대 순서 관계: {len(seq_rows)}개")

    # 국적-시대 연결
    nat_era_rows = []
    for nat, eras in NATIONALITY_ERAS.items():
        for era in eras:
            nat_era_rows.append({"nat": nat, "era": era})
    session.run("""
        UNWIND $rows AS row
        MERGE (n:국적 {name: row.nat})
        MERGE (s:시대 {name: row.era})
        MERGE (n)-[:시대포함]->(s)
    """, rows=nat_era_rows)
    print(f"   국적-시대 연결: {len(nat_era_rows)}개")


# ── 통계 출력 ─────────────────────────────────────────────────────────────────

def print_stats(session):
    print("\n[통계] 최종 그래프 현황:")
    labels = ["유물", "국적", "시대", "재질", "분류", "전시위치", "작가"]
    for label in labels:
        cnt = session.run(f"MATCH (n:{label}) RETURN count(n) AS c").single()["c"]
        print(f"   {label:6s} 노드: {cnt:,}개")

    rels = [
        ("유물", "국적",     "국적"),
        ("유물", "시대",     "속한시대"),
        ("유물", "재질",     "재질로만들어짐"),
        ("유물", "분류",     "분류됨"),
        ("유물", "전시위치", "전시위치"),
        ("유물", "작가",     "만든작가"),
        ("분류", "분류",     "상위분류"),
        ("시대", "시대",     "다음시대"),
        ("국적", "시대",     "시대포함"),
    ]
    print()
    for src, dst, rel in rels:
        cnt = session.run(
            f"MATCH (:{src})-[r:{rel}]->(:{dst}) RETURN count(r) AS c"
        ).single()["c"]
        print(f"   [{rel}] {cnt:,}개")


# ── 메인 ─────────────────────────────────────────────────────────────────────

def main():
    # 데이터 로드
    with open(DATA_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)
    print(f"[OK] 데이터 로드: {len(data):,}개 유물\n")

    # Neo4j 연결
    print(f"[연결] Neo4j 연결 중... ({NEO4J_URI})")
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    driver.verify_connectivity()
    print("[OK] Neo4j 연결 성공!\n")

    with driver.session() as session:

        # 제약조건
        print("[*] 인덱스/제약조건 생성 중...")
        create_constraints(session)
        print("[OK] 완료\n")

        # 유물 + 기본 관계 배치 삽입
        total = len(data)
        print(f"[>>] 유물 삽입 시작 (배치 크기: {BATCH_SIZE})\n")
        for i in range(0, total, BATCH_SIZE):
            batch = data[i : i + BATCH_SIZE]
            session.execute_write(insert_batch, batch)
            done = min(i + BATCH_SIZE, total)
            filled = done * 30 // total
            bar = "#" * filled + "-" * (30 - filled)
            print(f"\r  [{bar}] {done:,}/{total:,}", end="", flush=True)
        print("\n[OK] 유물 삽입 완료\n")

        # 강화 관계 구축
        print("[연결] 강화 관계 구축 중...")
        build_category_hierarchy(session, data)
        build_era_metadata(session)
        print("[OK] 강화 관계 완료\n")

        # 통계
        print_stats(session)

    driver.close()
    print("\n[완료] 그래프 DB 구축 완료!")


if __name__ == "__main__":
    main()
