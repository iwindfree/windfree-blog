---
title: "RAG 기반 고객 상담 챗봇 만들기"
author: iwindfree
pubDatetime: 2025-02-13T09:00:00Z
slug: "llm-chatbot-langchain"
category: "LLM Engineering"
series: "LLM Engineering"
seriesOrder: 18
tags: ["ai", "llm", "rag", "chatbot"]
description: "이 노트북에서는 LangChain과 ChromaDB를 사용하여 RAGRetrieval-Augmented Generation 기반의 고객 상담 챗봇을 구축하는 방법을 배웁니다."
---

이 노트북에서는 LangChain과 ChromaDB를 사용하여 **RAG(Retrieval-Augmented Generation)** 기반의 고객 상담 챗봇을 구축하는 방법을 배웁니다.

## 학습 목표

| 주제 | 설명 |
|------|------|
| 문서 로딩 | 마크다운 파일을 LangChain Document로 로드 |
| 청킹 | 문서를 적절한 크기로 분할하는 전략 |
| 임베딩 | 텍스트를 벡터로 변환하여 저장 |
| 검색 | 질문과 유사한 문서 검색 |
| RAG 체인 | 검색된 문서를 활용한 LLM 답변 생성 |
| 웹 UI | Gradio로 대화형 인터페이스 구축 |

---

## 1. 프로젝트 구조

```
06_using_rag
    example/
    ├── app.py                    # Gradio 웹 애플리케이션
    ├── src/
    │   ├── embed_documents.py    # 문서 임베딩 모듈
    │   └── rag_chain.py          # RAG 체인 모듈
    └── chroma_db/                # 벡터 DB (자동 생성)
00_test_data
    ├── knowledge_base/           # 지식베이스 문서
    │   ├── products/             # 제품 정보
    │   ├── policies/             # 정책 문서 (배송, 교환/환불, 보증)
    │   └── faq/                  # FAQ
    │   └── ...                   # 기타 문서들

```
### 데이터 흐름

```
1. 문서 준비 (knowledge_base/)
       ↓
2. 문서 로드 및 청킹 (embed_documents.py)
       ↓
3. 임베딩 & 벡터 DB 저장 (chroma_db/)
       ↓
4. 사용자 질문 → 유사 문서 검색 (rag_chain.py)
       ↓
5. 검색 결과 + 질문 → LLM 답변 생성
       ↓
6. Gradio UI로 응답 표시 (app.py)
```

## 2. 환경 설정


```python
# 필요한 패키지 (이미 설치되어 있다면 생략)
# !pip install langchain langchain-openai langchain-chroma langchain-huggingface langchain-community gradio python-dotenv
```



```python
import os
from pathlib import Path
from dotenv import load_dotenv

# 환경변수 로드
load_dotenv(override=True)

# API 키 확인
api_key = os.getenv("OPENAI_API_KEY")
if api_key:
    print(f"✅ OpenAI API 키 로드됨: {api_key[:8]}...")
else:
    print("❌ OPENAI_API_KEY가 설정되지 않았습니다. .env 파일을 확인하세요.")
```



<div class="nb-output">

```text
✅ OpenAI API 키 로드됨: sk-proj-...
```

</div>



```python
# 경로 설정
EXAMPLE_DIR = Path("./example")
#CURRENT_DIR = Path(__file__).parent 
CURRENT_DIR = Path.cwd()
BASE_DIR = CURRENT_DIR.parent
KNOWLEDGE_BASE_DIR = BASE_DIR / "00_test_data" / "knowledge_base"
CHROMA_DB_DIR = EXAMPLE_DIR / "chroma_db"

print(f"지식베이스: {KNOWLEDGE_BASE_DIR.absolute()}")
```



<div class="nb-output">

```text
지식베이스: /Users/windfree/workspace/ws.study/ai-engineering/llm_engineering/00_test_data/knowledge_base
```

</div>


## 3. 지식베이스 문서 살펴보기

우리가 준비한 지식베이스에는 가상의 쇼핑몰 "TechMall"의 정보가 들어있습니다.


```python
# 지식베이스 구조 확인
print("📁 지식베이스 구조:\n")

for category_dir in sorted(KNOWLEDGE_BASE_DIR.iterdir()):
    if category_dir.is_dir():
        files = list(category_dir.glob("*.md"))
        print(f"📂 {category_dir.name}/")
        for f in files:
            print(f"   └─ {f.name}")
```



<div class="nb-output">

```text
📁 지식베이스 구조:

📂 company/
   └─ overview.md
   └─ history.md
📂 contracts/
   └─ samsung_incentive.md
   └─ hyundai_mice.md
📂 employees/
   └─ kang_jihoon.md
   └─ song_minji.md
   └─ kwon_nara.md
   └─ kim_haneul.md
   └─ park_minsoo.md
📂 faq/
   └─ product.md
   └─ general.md
📂 policies/
   └─ cancellation.md
   └─ booking.md
   └─ insurance.md
📂 products/
   └─ vietnam_danang.md
   └─ japan_osaka.md
   └─ jeju_healing.md
   └─ maldives_honeymoon.md
```

</div>



```python
# 샘플 문서 내용 확인
sample_file = KNOWLEDGE_BASE_DIR / "products" / "jeju_healing.md"

print(f"📄 {sample_file.name} 내용:\n")
print(sample_file.read_text(encoding="utf-8")[:1000])
```



<div class="nb-output">

```text
📄 jeju_healing.md 내용:

# 제주 힐링 3박4일

## 상품 개요

- **상품명**: 제주 힐링 3박4일
- **상품코드**: JJ-HEAL-3N4D
- **여행 기간**: 3박 4일
- **출발지**: 김포/김해/청주/대구
- **도착지**: 제주국제공항
- **이동 수단**: 국내선 항공 + 전용 차량
- **가이드**: 전문 가이드 동행 (20명 이상 시)
- **최소 출발 인원**: 10명

---

## 가격 안내

### 기본 가격 (1인 기준)

| 시즌 | 성인 | 아동 (만 12세 미만) | 유아 (만 2세 미만) |
|------|------|-------------------|------------------|
| 비수기 (3-5월, 9-11월) | 549,000원 | 449,000원 | 50,000원 |
| 성수기 (7-8월, 12-2월) | 749,000원 | 649,000원 | 50,000원 |
| 연휴 (설/추석/황금연휴) | 849,000원 | 749,000원 | 50,000원 |

### 객실 추가 옵션

| 객실 타입 | 추가 비용 (1박 기준) |
|-----------|---------------------|
| 스탠다드 트윈 | 포함 |
| 디럭스 더블 | +30,000원 |
| 오션뷰 | +50,000원 |
| 스위트 | +120,000원 |
| 싱글룸 사용 | +90,000원/박 |

---

## 포함 사항

- 왕복 항공권 (위탁 수하물 15kg)
- 3박 숙박 (4성급 호텔 기준)
- 전 일정 전용 차량
- 전문 가이드 (20명 이상)
- 조식 3회
- 관광지 입장료 (성산일출봉, 만장굴, 천지연폭포)
- 여행자 보험 (기본)

---
... (출력 18줄 생략)
```

</div>


## 4. 문서 로딩 (Document Loading)

LangChain의 `DirectoryLoader`를 사용하여 마크다운 파일들을 로드합니다.


```python
from langchain_community.document_loaders import DirectoryLoader, TextLoader

def load_all_documents(base_path: Path) -> list:
    """
    지식베이스의 모든 마크다운 문서를 로드합니다.
    각 문서에 카테고리 메타데이터를 추가합니다.
    """
    documents = []
    
    for category_folder in base_path.iterdir():
        if not category_folder.is_dir():
            continue
        
        category = category_folder.name
        
        # DirectoryLoader로 해당 폴더의 .md 파일 로드
        loader = DirectoryLoader(
            str(category_folder),
            glob="**/*.md",
            loader_cls=TextLoader,
            loader_kwargs={"encoding": "utf-8"}
        )
        
        docs = loader.load()
        
        # 메타데이터에 카테고리, 파일이름 정보  추가 제공
        for doc in docs:
            doc.metadata["category"] = category
            doc.metadata["filename"] = Path(doc.metadata["source"]).name
            documents.append(doc)
    
    return documents
```



```python
# 문서 로드
documents = load_all_documents(KNOWLEDGE_BASE_DIR)

print(f"총 {len(documents)}개 문서 로드됨\n")

# 로드된 문서 정보 출력
for doc in documents:
    content_preview = doc.page_content[:50].replace("\n", " ") + "..."
    print(f"📄 [{doc.metadata['category']}] {doc.metadata['filename']}")
    print(f"   내용: {content_preview}")
    print(f"   길이: {len(doc.page_content):,}자\n")
```



<div class="nb-output">

```text
총 18개 문서 로드됨

📄 [products] vietnam_danang.md
   내용: # 베트남 다낭 4박5일  ## 상품 개요  - **상품명**: 베트남 다낭-호이안 4박5...
   길이: 3,999자

📄 [products] japan_osaka.md
   내용: # 일본 오사카-도쿄 5박6일  ## 상품 개요  - **상품명**: 일본 오사카-도쿄 골...
   길이: 4,014자

📄 [products] jeju_healing.md
   내용: # 제주 힐링 3박4일  ## 상품 개요  - **상품명**: 제주 힐링 3박4일 - **...
   길이: 2,944자

📄 [products] maldives_honeymoon.md
   내용: # 몰디브 허니문 6박7일  ## 상품 개요  - **상품명**: 몰디브 워터빌라 허니문 ...
   길이: 3,732자

📄 [faq] product.md
   내용: # 여행상품 자주 묻는 질문 (FAQ)  ## 예약 및 결제  ### Q1: 예약은 얼마나...
   길이: 5,063자

📄 [faq] general.md
   내용: # 일반 자주 묻는 질문 (FAQ)  ## 회원가입 및 계정  ### Q1: 회원가입은 어...
   길이: 3,907자

📄 [contracts] samsung_incentive.md
   내용: # 삼성전자 인센티브 투어 계약  ## 계약 개요  - **계약명**: 삼성전자 인센티브 ...
   길이: 2,634자

📄 [contracts] hyundai_mice.md
   내용: # 현대자동차 MICE 계약  ## 계약 개요  - **계약명**: 현대자동차 MICE 행...
   길이: 3,017자

📄 [policies] cancellation.md
   내용: # 취소 및 환불 정책  ## 하늘여행사 취소/환불 규정  ### 개요  하늘여행사는 고객...
   길이: 2,407자

📄 [policies] booking.md
   내용: # 예약 및 출발 정책  ## 하늘여행사 예약 안내  ### 예약 방법  #### 1. 온...
   길이: 3,108자

📄 [policies] insurance.md
   내용: # 여행자 보험 안내  ## 하늘여행사 여행자 보험  ### 개요  하늘여행사는 고객의 안...
   길이: 2,698자

📄 [company] overview.md
   내용: # 회사 개요  ## 주식회사 하늘여행사  ### 기본 정보  - **회사명**: 주식회사...
   길이: 1,486자

... (출력 23줄 생략)
```

</div>


## 5. 청킹 (Chunking)

문서가 너무 길면 검색 효율이 떨어집니다. 적절한 크기로 분할해야 합니다.

### 청킹 전략

| 파라미터 | 설명 | 권장값 |
|---------|------|-------|
| chunk_size | 청크의 최대 문자 수 | 300-1000 |
| chunk_overlap | 청크 간 겹치는 문자 수 | chunk_size의 10-20% |
| separators | 분할 기준 문자열 | 헤더, 단락 순서로 |

### RecursiveCharacterTextSplitter

가장 많이 사용되는 텍스트 분할기입니다. 지정된 separators 순서대로 분할을 시도합니다.


```python
from langchain_text_splitters import RecursiveCharacterTextSplitter

# 텍스트 분할기 설정
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100,
    separators=["\n## ", "\n### ", "\n\n", "\n", " ", ""]
)

# 문서 분할
chunks = text_splitter.split_documents(documents)

print(f"원본 문서 수: {len(documents)}")
print(f"청크 수: {len(chunks)}")
print(f"평균 청크 크기: {sum(len(c.page_content) for c in chunks) / len(chunks):.0f}자")
```



<div class="nb-output">

```text
원본 문서 수: 18
청크 수: 156
평균 청크 크기: 331자
```

</div>



```python
# 청크 샘플 확인
print("=" * 60)
print("청크 샘플 (처음 3개)")
print("=" * 60)

for i, chunk in enumerate(chunks[:3]):
    print(f"\n--- 청크 {i+1} ---")
    print(f"카테고리: {chunk.metadata['category']}")
    print(f"파일: {chunk.metadata['filename']}")
    print(f"길이: {len(chunk.page_content)}자")
    print(f"내용:\n{chunk.page_content[:300]}...")
```



<div class="nb-output">

```text
============================================================
청크 샘플 (처음 3개)
============================================================

--- 청크 1 ---
카테고리: products
파일: vietnam_danang.md
길이: 211자
내용:
# 베트남 다낭 4박5일

## 상품 개요

- **상품명**: 베트남 다낭-호이안 4박5일
- **상품코드**: VN-DNG-4N5D
- **여행 기간**: 4박 5일
- **출발지**: 인천국제공항
- **도착지**: 다낭 국제공항
- **이동 수단**: 국제선 항공 + 전용 차량
- **가이드**: 한국어 가이드 전 일정 동행
- **최소 출발 인원**: 15명

---...

--- 청크 2 ---
카테고리: products
파일: vietnam_danang.md
길이: 428자
내용:
## 가격 안내

### 기본 가격 (1인 기준)

| 시즌 | 성인 | 아동 (만 12세 미만) | 유아 (만 2세 미만) |
|------|------|-------------------|------------------|
| 비수기 (5-9월) | 890,000원 | 790,000원 | 150,000원 |
| 성수기 (10-4월) | 1,190,000원 | 1,050,000원 | 150,000원 |
| 연휴 (설/추석) | 1,390,000원 | 1,250,000원 | 150,000원 |

### 객실 추가 옵션

| 객실 타...

--- 청크 3 ---
카테고리: products
파일: vietnam_danang.md
길이: 149자
내용:
### 호텔 업그레이드

... (출력 7줄 생략)
```

</div>


## 6. 임베딩 & 벡터 저장 (Embedding & Vector Store)

청크를 벡터로 변환하여 ChromaDB에 저장합니다.

### 임베딩 모델 선택

| 모델 | 차원 | 특징 |
|------|-----|------|
| all-MiniLM-L6-v2 | 384 | 무료, 빠름, 로컬 실행 |
| text-embedding-3-small | 1536 | OpenAI, 유료, 고성능 |
| text-embedding-3-large | 3072 | OpenAI, 유료, 최고 성능 |

이번 예제에서는 무료로 로컬에서 실행 가능한 HuggingFace 모델을 사용합니다.


```python
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
import shutil

# 임베딩 모델 초기화
EMBEDDING_MODEL_OPENAI = "text-embedding-3-small"

print(f"임베딩 모델 로드 중: {EMBEDDING_MODEL_OPENAI}")
embedding_model = OpenAIEmbeddings(model=EMBEDDING_MODEL_OPENAI)
print("✅ 임베딩 모델 로드 완료")
```



<div class="nb-output">

```text
임베딩 모델 로드 중: text-embedding-3-small
✅ 임베딩 모델 로드 완료
```

</div>



```python
# 임베딩 테스트
test_text = "제주도 여행"
test_embedding = embedding_model.embed_query(test_text)

print(f"테스트 텍스트: '{test_text}'")
print(f"임베딩 차원: {len(test_embedding)}")
print(f"임베딩 샘플: {test_embedding[:5]}")
```



<div class="nb-output">

```text
테스트 텍스트: '제주도 여행'
임베딩 차원: 1536
임베딩 샘플: [0.035603079944849014, -0.004754707217216492, -0.024595465511083603, 0.03015129268169403, 0.021370170637965202]
```

</div>



```python
# 기존 DB 삭제 (재생성을 위해)
if CHROMA_DB_DIR.exists():
    shutil.rmtree(CHROMA_DB_DIR)
    print("기존 벡터 DB 삭제됨")

# 벡터 스토어 생성
print("\n벡터 DB 생성 중...")

vector_store = Chroma.from_documents(
    documents=chunks,
    embedding=embedding_model,
    persist_directory=str(CHROMA_DB_DIR)
)

# 저장 정보 확인
collection = vector_store._collection
print(f"\n✅ 벡터 DB 생성 완료")
print(f"   저장된 벡터 수: {collection.count():,}개")
print(f"   저장 경로: {CHROMA_DB_DIR}")
```



<div class="nb-output">

```text
기존 벡터 DB 삭제됨

벡터 DB 생성 중...

✅ 벡터 DB 생성 완료
   저장된 벡터 수: 156개
   저장 경로: example/chroma_db
```

</div>


## 7. 유사 문서 검색 (Retrieval)

벡터 DB에서 질문과 유사한 문서를 검색합니다.


```python
# Retriever 생성
retriever = vector_store.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 5}  # 상위 5개 문서 반환
)

# 검색 테스트
test_query = "회사 창립 연도가 언제야?"
retrieved_docs = retriever.invoke(test_query)

print(f"질문: {test_query}")
print(f"\n검색된 문서 수: {len(retrieved_docs)}\n")
print("=" * 60)

for i, doc in enumerate(retrieved_docs, 1):
    print(f"\n[{i}] {doc.metadata['category']} / {doc.metadata['filename']}")
    print(f"{doc.page_content[:200]}...")
```



<div class="nb-output">

```text
질문: 회사 창립 연도가 언제야?

검색된 문서 수: 5

============================================================

[1] company / overview.md
# 회사 개요...

[2] company / history.md
# 회사 연혁...

[3] company / history.md
## 주식회사 하늘여행사 연혁
### 2008년 - 창립

- **3월 15일**: 주식회사 하늘여행사 설립 (자본금 5천만원)
- **3월 20일**: 서울 중구 을지로 첫 사무실 개설
- **4월**: 문화체육관광부 일반여행업 등록 완료
- **5월**: 첫 패키지 상품 "제주도 3박4일" 출시
- **12월**: 창립 첫해 고객 1,200명 달성

#...

[4] company / overview.md
## 주식회사 하늘여행사

### 기본 정보

- **회사명**: 주식회사 하늘여행사
- **영문명**: Sky Travel Co., Ltd.
- **설립일**: 2008년 3월 15일
- **창립자**: 김하늘
- **대표이사**: 김하늘
- **본사 주소**: 서울특별시 중구 명동길 55, 하늘빌딩 7층
- **대표 전화**: 02-1234-5678
-...

[5] company / history.md
### 2024년 - 현재

- **1월**: 삼성전자 인센티브 투어 계약 갱신 (2년, 연 20억원)
- **3월 15일**: 창립 16주년
- **4월**: 직원 45명 달성
- **5월**: 현대자동차 MICE 계약 갱신 (1년, 연 15억원)
- **8월**: AI 챗봇 고객 상담 서비스 도입
- **10월**: 연간 고객 50,000명 조기 달성
...
```

</div>


## 8. RAG 체인 구현

검색된 문서를 바탕으로 LLM이 답변을 생성하도록 합니다.


```python
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

# LLM 초기화
llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.1
)

# 시스템 프롬프트 템플릿
SYSTEM_PROMPT = """당신은 하늘 여행사의 친절한 고객 상담원입니다.
고객의 질문에 정확하고 도움이 되는 답변을 제공하세요.

## 답변 지침
- 아래 참고 자료를 기반으로 답변하세요.
- 참고 자료에 없는 내용은 "확인 후 안내드리겠습니다"라고 말씀해주세요.
- 친절하고 전문적인 톤을 유지하세요.

## 참고 자료
{context}
"""

print("LLM 준비 완료")
```



<div class="nb-output">

```text
LLM 준비 완료
```

</div>



```python
def ask_with_rag(question: str) -> str:
    """
    RAG 파이프라인으로 질문에 답변합니다.
    """
    # 1. 관련 문서 검색
    docs = retriever.invoke(question)
    
    # 2. 컨텍스트 구성
    context_parts = []
    for doc in docs:
        category = doc.metadata.get("category", "")
        context_parts.append(f"[{category}]\n{doc.page_content}")
    context = "\n\n---\n\n".join(context_parts)
    
    # 3. 프롬프트 구성
    system_message = SYSTEM_PROMPT.format(context=context)
    
    # 4. LLM 호출
    messages = [
        SystemMessage(content=system_message),
        HumanMessage(content=question)
    ]
    
    response = llm.invoke(messages)
    return response.content
```



```python
# RAG 테스트
test_questions = [
    "제주도 관련 여행 상품을 알려줘요",
    "환불은 어떻게 신청하나요?",    
]

for q in test_questions:
    print(f"\n{'='*60}")
    print(f"Q: {q}")
    print(f"{'='*60}")
    answer = ask_with_rag(q)
    print(f"\nA: {answer}")
```



<div class="nb-output">

```text

============================================================
Q: 제주도 관련 여행 상품을 알려줘요
============================================================

A: 저희 하늘 여행사에서는 제주도 관련 여행 상품으로 "제주 힐링 3박4일" 상품을 제공하고 있습니다. 

### 제주 힐링 3박4일 상품 개요
- **상품명**: 제주 힐링 3박4일
- **상품코드**: JJ-HEAL-3N4D
- **여행 기간**: 3박 4일
- **출발지**: 김포/김해/청주/대구
- **도착지**: 제주국제공항
- **이동 수단**: 국내선 항공 + 전용 차량
- **가이드**: 전문 가이드 동행 (20명 이상 시)
- **최소 출발 인원**: 10명

이 외에도 다양한 옵션 투어가 준비되어 있으니, 필요하신 경우 추가 정보를 요청해 주세요!

============================================================
Q: 환불은 어떻게 신청하나요?
============================================================

A: 환불 신청은 다음과 같은 방법으로 가능합니다:

1. **전화**: 02-1234-5678로 연락 주시면 됩니다. (평일 09:00-18:00)
2. **이메일**: cancel@skytravel.co.kr로 신청하실 수 있습니다.
3. **카카오톡**: @하늘여행사 채널을 통해 문의하실 수 있습니다.
4. **방문**: 각 지사 영업시간 내에 직접 방문하셔도 됩니다.

환불 신청 시 필요한 서류는 다음과 같습니다:
- 예약 확인서 또는 예약번호
- 예약자 신분증 사본
- 환불 계좌 정보 (계좌이체 환불 시)

추가로 궁금한 점이 있으시면 언제든지 문의해 주세요!
```

</div>


## 9. RAG vs 일반 LLM 비교

RAG를 사용하지 않으면 어떻게 될까요?


```python
def ask_without_rag(question: str) -> str:
    """RAG 없이 LLM에 직접 질문"""
    messages = [
        SystemMessage(content="당신은 하늘여행사의 고객 상담원입니다."),
        HumanMessage(content=question)
    ]
    response = llm.invoke(messages)
    return response.content

# 비교
comparison_question = "제주도 여행 상품 정보를 말해줘"

print("질문:", comparison_question)
print("\n" + "="*60)
print("📚 RAG 사용 (지식베이스 참조)")
print("="*60)
print(ask_with_rag(comparison_question))

print("\n" + "="*60)
print("❌ RAG 미사용 (LLM 지식만)")
print("="*60)
print(ask_without_rag(comparison_question))
```



<div class="nb-output">

```text
질문: 제주도 여행 상품 정보를 말해줘

============================================================
📚 RAG 사용 (지식베이스 참조)
============================================================
저희 하늘 여행사에서 제공하는 제주도 여행 상품은 "제주 힐링 3박4일"입니다. 아래는 상품의 주요 정보입니다.

- **상품명**: 제주 힐링 3박4일
- **상품코드**: JJ-HEAL-3N4D
- **여행 기간**: 3박 4일
- **출발지**: 김포, 김해, 청주, 대구
- **도착지**: 제주국제공항
- **이동 수단**: 국내선 항공 + 전용 차량
- **가이드**: 전문 가이드 동행 (20명 이상 시)
- **최소 출발 인원**: 10명

이 외에도 다양한 옵션 투어가 준비되어 있으니, 추가적인 정보가 필요하시거나 궁금한 점이 있으시면 언제든지 문의해 주세요!

============================================================
❌ RAG 미사용 (LLM 지식만)
============================================================
제주도 여행 상품은 다양하게 제공되고 있습니다. 일반적으로 포함되는 내용은 다음과 같습니다:

1. **항공권**: 제주도까지의 왕복 항공권이 포함됩니다. 출발지는 서울, 부산, 인천 등 다양한 선택지가 있습니다.

2. **숙박**: 호텔, 리조트, 펜션 등 다양한 숙박 옵션이 제공됩니다. 숙소의 위치와 가격대에 따라 선택할 수 있습니다.

3. **렌터카**: 제주도는 대중교통이 제한적이므로 렌터카 서비스가 포함된 상품이 많습니다. 자유롭게 관광할 수 있는 장점이 있습니다.

4. **관광지 투어**: 제주도의 주요 관광지(한라산, 성산일출봉, 만장굴, 우도 등)를 포함한 패키지 투어가 제공됩니다. 가이드와 함께하는 옵션도 있습니다.

5. **식사**: 제주도의 특산물인 흑돼지, 해산물 등을 포함한 식사가 제공되는 상품도 있습니다.

6. **액티비티**: 스노클링, 서핑, ATV 체험 등 다양한 액티비티를 추가할 수 있는 옵션도 있습니다.

여행 일정, 가격, 포함 사항은 여행사마다 다를 수 있으니, 구체적인 상품에 대한 정보는 하늘여행사 웹사이트나 고객센터를 통해 확인해 주시기 바랍니다. 추가로 궁금한 사항이 있으시면 언제든지 문의해 주세요!
```

</div>


## 10. Gradio UI로 챗봇 실행

이제 완성된 RAG 챗봇을 Gradio UI로 실행해봅시다.

터미널에서 다음 명령을 실행하세요:

```bash
cd basic/day6/example
python app.py
```

또는 아래 셀을 실행하여 간단한 챗 인터페이스를 띄울 수 있습니다.


```python
import gradio as gr

def chat_fn(message, history):
    """Gradio용 채팅 함수"""
    return ask_with_rag(message)

# 간단한 채팅 UI
demo = gr.ChatInterface(
    fn=chat_fn,
    title="하늘여행사 고객 상담 챗봇",
    description="여행 상품, 예약, 환불 등에 대해 물어보세요!",
    examples=[
        "제주도 관련 여행 상품을 알려줘요",
        "환불은 어떻게 신청하나요?",
    ]
)

demo.launch()
```



<div class="nb-output">

```text
* Running on local URL:  http://127.0.0.1:7860
* To create a public link, set `share=True` in `launch()`.
<IPython.core.display.HTML object>
```

</div>


## 11. 요약

### RAG 파이프라인 구성요소

| 단계 | 모듈 | 역할 |
|------|------|------|
| 1. 문서 로딩 | DirectoryLoader | 파일 시스템에서 문서 읽기 |
| 2. 청킹 | RecursiveCharacterTextSplitter | 문서를 적절한 크기로 분할 |
| 3. 임베딩 | HuggingFaceEmbeddings | 텍스트를 벡터로 변환 |
| 4. 저장 | ChromaDB | 벡터 저장 및 검색 |
| 5. 검색 | Retriever | 유사 문서 검색 |
| 6. 생성 | ChatOpenAI | 컨텍스트 기반 답변 생성 |
| 7. UI | Gradio | 대화형 웹 인터페이스 |

### 핵심 포인트

1. **청킹 전략이 중요**: chunk_size와 overlap을 문서 특성에 맞게 조정
2. **메타데이터 활용**: 카테고리, 출처 등을 저장하여 필터링/디버깅에 활용
3. **프롬프트 엔지니어링**: 시스템 프롬프트로 LLM의 동작 방식 제어
4. **RAG의 장점**: 최신 정보 반영, 환각 감소, 출처 추적 가능

### 다음 단계

- 하이브리드 검색 (BM25 + 벡터)
- Re-ranking으로 검색 품질 향상
- 스트리밍 응답
- 대화 이력 관리
