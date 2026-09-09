# Museum Relics Chroma Vector DB README

## 1. 개요

본 벡터 DB는 국립중앙박물관 유물 JSON 데이터 2,560건을 대상으로 ChromaDB를 이용해 구축한 로컬 벡터 데이터베이스입니다.

각 유물 데이터는 다음 6개 필드를 하나의 문서 텍스트로 결합한 뒤 임베딩했습니다.

- `relicId`
- `title`
- `다른명칭`
- `전시명칭`
- `소장품번호`
- `description`

임베딩 모델은 SentenceTransformer 기반 한국어 모델인 `jhgan/ko-sroberta-multitask`를 사용했습니다.

---

## 2. DB 기본 정보

| 항목 | 값 |
|---|---|
| DB 종류 | ChromaDB |
| 저장 방식 | Local PersistentClient |
| DB 폴더명 | `chroma_museum_db` |
| 컬렉션 이름 | `museum_relics` |
| 저장 유물 수 | 2,560개 |
| 임베딩 모델 | `jhgan/ko-sroberta-multitask` |
| 임베딩 차원 | 768차원 |
| 내부 ID 형식 | `relic_{relicId}` |

---

## 3. DB 파일 구조

Chroma DB는 폴더 전체를 유지해야 정상적으로 작동합니다.

예상 폴더 구조는 다음과 같습니다.

```text
chroma_museum_db/
├─ chroma.sqlite3
└─ <uuid 형태의 index 폴더>/
   ├─ data_level0.bin
   ├─ header.bin
   ├─ index_metadata.pickle
   ├─ length.bin
   └─ link_lists.bin
```

각 파일의 역할은 다음과 같습니다.

| 파일/폴더 | 설명 |
|---|---|
| `chroma.sqlite3` | 문서, metadata, 컬렉션 정보 저장 |
| `data_level0.bin` | HNSW 벡터 인덱스 데이터 |
| `header.bin` | HNSW 인덱스 헤더 정보 |
| `index_metadata.pickle` | 인덱스 metadata |
| `length.bin` | 벡터 길이 및 관련 정보 |
| `link_lists.bin` | HNSW 그래프 링크 정보 |

주의: DB를 이동하거나 제출할 때는 `chroma.sqlite3`만 옮기면 안 됩니다. 반드시 `chroma_museum_db` 폴더 전체를 압축해서 옮겨야 합니다.

---

## 4. 임베딩에 사용한 필드

각 유물은 아래와 같은 텍스트 형식으로 결합된 뒤 임베딩되었습니다.

```text
유물ID: 329
유물명: 흑갈유 병
다른명칭: 黑褐釉甁
전시명칭: 흑유 철반무늬 병
소장품번호: 본관215
설명문: 이 유물은 얕은 입 둘레, 짧고 가는 목...
```

즉, 단순히 `description`만 임베딩한 것이 아니라, 유물명, 다른 명칭, 전시 명칭, 소장품 번호, 설명문, 유물 ID까지 함께 임베딩한 구조입니다.

---

## 5. 저장된 metadata 구조

각 유물의 metadata에는 다음 필드가 저장되어 있습니다.

```python
{
    "relicId": 329,
    "title": "흑갈유 병",
    "다른명칭": "黑褐釉甁",
    "전시명칭": "흑유 철반무늬 병",
    "소장품번호": "본관215",
    "description": "이 유물은 얕은 입 둘레..."
}
```

Chroma 내부적으로는 문서 원문을 위한 `chroma:document`도 함께 저장됩니다.

따라서 전체 metadata row는 다음과 같이 계산됩니다.

```text
2,560개 유물 × 7개 항목 = 17,920개 metadata row
```

7개 항목은 다음과 같습니다.

```text
chroma:document
relicId
title
다른명칭
전시명칭
소장품번호
description
```

---

## 6. 내부 ID 구조

Chroma 내부 ID는 다음 형식으로 저장했습니다.

```text
relic_{relicId}
```

예시:

```text
relicId: 329
Chroma ID: relic_329
```

이 구조를 사용하면 검색 결과가 어떤 원본 유물과 연결되는지 쉽게 추적할 수 있습니다.

---

## 7. DB 접속 코드

아래 코드를 사용하면 구축된 Chroma DB에 접속할 수 있습니다.

```python
import chromadb
from chromadb.utils import embedding_functions

DB_DIR = "chroma_museum_db"
COLLECTION_NAME = "museum_relics"

embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="jhgan/ko-sroberta-multitask"
)

client = chromadb.PersistentClient(path=DB_DIR)

collection = client.get_collection(
    name=COLLECTION_NAME,
    embedding_function=embedding_function
)

print("저장 문서 수:", collection.count())
```

정상적으로 구축되었다면 다음과 같이 출력됩니다.

```text
저장 문서 수: 2560
```

---

## 8. 검색 테스트 코드

```python
import chromadb
from chromadb.utils import embedding_functions

DB_DIR = "chroma_museum_db"
COLLECTION_NAME = "museum_relics"

embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="jhgan/ko-sroberta-multitask"
)

client = chromadb.PersistentClient(path=DB_DIR)

collection = client.get_collection(
    name=COLLECTION_NAME,
    embedding_function=embedding_function
)

query = "고려시대 청자 유물"

results = collection.query(
    query_texts=[query],
    n_results=5
)

for i in range(len(results["ids"][0])):
    meta = results["metadatas"][0][i]

    print("=" * 80)
    print("순위:", i + 1)
    print("거리:", results["distances"][0][i])
    print("Chroma ID:", results["ids"][0][i])
    print("relicId:", meta["relicId"])
    print("유물명:", meta["title"])
    print("다른명칭:", meta["다른명칭"])
    print("전시명칭:", meta["전시명칭"])
    print("소장품번호:", meta["소장품번호"])
    print("설명:", meta["description"][:300])
```

---

## 9. 사용한 라이브러리 버전

DB 구축에 사용한 실행 환경과 주요 라이브러리 버전은 다음과 같습니다.

| 항목 | 버전 |
|---|---|
| Python | `3.10.20` |
| Python 배포판 | `Anaconda, Inc.` |
| Python 빌드 | `main, Mar 11 2026, 17:42:35` |
| 컴파일러 | `MSC v.1942 64 bit (AMD64)` |
| ChromaDB | `1.5.9` |
| sentence-transformers | `5.5.1` |
| transformers | `5.8.1` |
| torch | `2.12.0+cpu` |

전체 패키지 목록을 별도로 보관하려면 터미널에서 다음 명령어를 실행하면 됩니다.

```bash
pip freeze > requirements.txt
```

---

## 10. requirements.txt 생성 권장

DB를 다른 환경에서 다시 실행하거나 공유하려면 `requirements.txt` 파일을 함께 저장하는 것이 좋습니다.

예시 명령어:

```bash
conda activate museum_chroma
pip freeze > requirements.txt
```

최소 필요 패키지는 다음과 같습니다.

```text
chromadb
sentence-transformers
torch
transformers
tqdm
```

---

## 11. DB 재구축에 사용한 핵심 설정

| 항목 | 설정 |
|---|---|
| JSON 파일 | `museum_collections.json` |
| DB 저장 폴더 | `chroma_museum_db` |
| 컬렉션 이름 | `museum_relics` |
| 임베딩 모델 | `jhgan/ko-sroberta-multitask` |
| 배치 크기 | 100 |
| 저장 문서 수 | 2,560개 |
| 임베딩 대상 필드 | `relicId`, `title`, `다른명칭`, `전시명칭`, `소장품번호`, `description` |
| metadata 저장 필드 | `relicId`, `title`, `다른명칭`, `전시명칭`, `소장품번호`, `description` |

---

## 12. 최종 설명

본 벡터 DB는 국립중앙박물관 유물 JSON 데이터 2,560건을 대상으로 ChromaDB를 이용해 구축하였다. 각 유물은 `relicId`, `title`, `다른명칭`, `전시명칭`, `소장품번호`, `description` 필드를 하나의 문서 텍스트로 결합한 뒤, SentenceTransformer 기반 한국어 임베딩 모델인 `jhgan/ko-sroberta-multitask`를 사용하여 768차원 벡터로 변환하였다.

Chroma 컬렉션 이름은 `museum_relics`이며, 각 문서의 내부 ID는 `relic_{relicId}` 형식으로 저장하였다. metadata에는 `relicId`, `title`, `다른명칭`, `전시명칭`, `소장품번호`, `description`을 함께 저장하여 검색 결과에서 원본 유물 정보를 바로 확인할 수 있도록 구성하였다.

---

## 13. 주의사항

1. DB를 이동할 때는 `chroma_museum_db` 폴더 전체를 압축해서 이동해야 합니다.
2. `chroma.sqlite3` 파일만 이동하면 인덱스 파일이 누락되어 검색이 정상 작동하지 않을 수 있습니다.
3. 검색할 때는 DB 구축에 사용한 것과 동일한 임베딩 모델인 `jhgan/ko-sroberta-multitask`를 사용해야 합니다.
4. 다른 임베딩 모델을 사용하면 벡터 공간이 달라져 검색 결과가 왜곡될 수 있습니다.
5. 기존 DB를 삭제하고 재구축할 경우 컬렉션 이름이 동일하면 기존 컬렉션을 먼저 삭제해야 합니다.
