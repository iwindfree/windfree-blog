---
title: "고급 RAG: 벡터 데이터베이스를 활용한 문서 검색 시스템"
author: iwindfree
pubDatetime: 2025-02-12T09:00:00Z
slug: "llm-advanced-rag-vectordb"
category: "AI Engineering"
tags: ["ai", "llm", "rag", "vectordb"]
description: "이번 노트북에서는 11번 노트북에서 배운 RAG 개념을 확장하여, 실전에서 사용하는 고급 RAG 시스템을 구축합니다. 벡터 데이터베이스를 사용하여 대량의 문서를 효율적으로 검색하고, LLM이 정확한 답변을 생성하도록 만들어봅니다."
---

## 개요

이번 노트북에서는 11번 노트북에서 배운 RAG 개념을 확장하여, **실전에서 사용하는 고급 RAG 시스템**을 구축합니다. 벡터 데이터베이스를 사용하여 대량의 문서를 효율적으로 검색하고, LLM이 정확한 답변을 생성하도록 만들어봅니다.

| 주제 | 내용 |
|------|------|
| RAG 복습 | 11번에서 배운 간단한 RAG 복습 |
| 문서 청킹 | 대용량 문서를 작은 조각으로 분할 |
| 벡터 DB | ChromaDB를 사용한 벡터 저장소 구축 |
| 고급 검색 | 의미 기반 문서 검색 |
| 시각화 | t-SNE를 사용한 벡터 시각화 |

## 학습 목표

1. 키워드 검색과 벡터 검색의 차이 이해하기
2. LangChain을 사용한 문서 로딩 및 청킹 마스터하기
3. ChromaDB 벡터 스토어 구축 및 활용하기
4. 실전 RAG 시스템 구현하기
5. 벡터를 시각화하여 검색 원리 이해하기

## 실습 시나리오

**CloudStore**라는 클라우드 스토리지 제품의 문서 검색 시스템을 만듭니다. 사용자는 제품 기능, API 사용법, 보안 정책 등에 대해 질문하고, AI 어시스턴트가 관련 문서를 찾아 정확한 답변을 제공합니다.

---

## 0. 필요한 라이브러리 설치 및 임포트


```python
pip install openai langchain langchain-openai langchain-community langchain-chroma langchain-huggingface chromadb tiktoken numpy scikit-learn plotly python-dotenv
```



```python
import os
import glob
import tiktoken
import numpy as np
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI

# LangChain 관련
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# 시각화 관련
from sklearn.manifold import TSNE
import plotly.graph_objects as go
```



```python
# API 키 설정
load_dotenv(override=True)
openai_api_key = os.getenv('OPENAI_API_KEY')

if openai_api_key:
    print(f"✅ OpenAI API Key loaded (시작: {openai_api_key[:8]}...)")
else:
    print("❌ OpenAI API Key not found")

MODEL = "gpt-4"
client = OpenAI()
```



<div class="nb-output">

```text
✅ OpenAI API Key loaded (시작: sk-proj-...)
```

</div>


## 1. RAG 개념 복습

11번 노트북에서 우리는 간단한 RAG를 구현했습니다. 복습해봅시다:

### RAG의 기본 흐름

```
1. 질문 받기: "Pro 플랜 가격이 얼마인가요?"
2. 관련 문서 검색: pricing.md 문서 찾기
3. 컨텍스트 주입: 찾은 문서 + 질문을 LLM에 전달
4. 답변 생성: "Pro 플랜은 월 $15입니다."
```

### 11번에서 사용한 방법 vs 이번 노트북

| 비교 항목 | 11번 노트북 | 12번 노트북 |
|----------|------------|-------------|
| 검색 방식 | numpy 코사인 유사도 | ChromaDB 벡터 검색 |
| 문서 처리 | 작은 예제 텍스트 | 대용량 문서 청킹 |
| 확장성 | 소규모 | 대규모 프로덕션 |
| 임베딩 | OpenAI API | HuggingFace (무료) |
| 시각화 | 없음 | t-SNE 시각화 |

---

## 2. Part 1: 간단한 키워드 기반 RAG

먼저 가장 간단한 방식인 **키워드 매칭**으로 RAG를 구현해봅시다. 문서에서 질문의 키워드를 찾아 관련 문서를 검색하는 방식입니다.


```python
# 모든 문서를 딕셔너리로 로드
knowledge = {}

# sample_docs 폴더의 모든 .md 파일 읽기
doc_path = "sample_docs/**/*.md"
filenames = glob.glob(doc_path, recursive=True)

print(f"📚 {len(filenames)}개의 문서를 발견했습니다:\n")

for filename in filenames:
    # 파일명에서 키 생성 (예: cloudstore_pro.md -> cloudstore_pro)
    key = Path(filename).stem.lower()
    with open(filename, "r", encoding="utf-8") as f:
        knowledge[key] = f.read()
    print(f"  - {key}: {len(knowledge[key]):,} 문자")

print(f"\n✅ 총 {len(knowledge)}개의 문서가 로드되었습니다.")
```



<div class="nb-output">

```text
📚 9개의 문서를 발견했습니다:

  - cloudstore_pro: 657 문자
  - cloudstore_enterprise: 877 문자
  - cloudstore: 524 문자
  - pricing: 4,173 문자
  - quickstart: 2,893 문자
  - security: 4,045 문자
  - webhook_api: 4,342 문자
  - auth_api: 3,033 문자
  - storage_api: 1,886 문자

✅ 총 9개의 문서가 로드되었습니다.
```

</div>



```python
# 키워드 기반 검색 함수
def get_relevant_context_keyword(message):
    """질문에서 키워드를 추출하여 관련 문서 검색"""
    # 알파벳과 공백만 남기고 제거
    text = ''.join(ch for ch in message if ch.isalpha() or ch.isspace())
    words = text.lower().split()
    
    # 키워드가 문서명에 포함된 경우 해당 문서 반환
    relevant_docs = []
    for word in words:
        for doc_key, doc_content in knowledge.items():
            if word in doc_key and doc_content not in relevant_docs:
                relevant_docs.append(doc_content)
    
    return relevant_docs

# 테스트
test_query = "CloudStore Pro 플랜의 가격이 얼마인가요?"
results = get_relevant_context_keyword(test_query)

print(f"질문: {test_query}")
print(f"\n찾은 문서 개수: {len(results)}")
if results:
    print(f"첫 번째 문서 미리보기:\n{results[0][:200]}...")
```



<div class="nb-output">

```text
질문: CloudStore Pro 플랜의 가격이 얼마인가요?

찾은 문서 개수: 3
첫 번째 문서 미리보기:
# CloudStore Pro

## 개요

CloudStore Pro는 중소기업과 전문가를 위한 고급 클라우드 스토리지 솔루션입니다. 향상된 협업 기능과 관리 도구를 제공합니다.

## 주요 기능

### 스토리지 용량
- 사용자당 1TB 제공
- 팀 공유 스토리지: 추가 5TB
- 필요시 용량 확장 가능

### 고급 협업 기능
- 실시간 문서 공동 편...
```

</div>


### 키워드 검색의 한계

키워드 검색의 문제점:

1. **동의어 문제**: "가격"과 "요금"을 다르게 취급
2. **문맥 무시**: 단어의 의미를 이해하지 못함
3. **정확한 매칭 필요**: 오타나 변형에 취약

예시:
- "Pro 플랜 비용은?" → "pro" 키워드로 찾음 ✅
- "프로 버전 얼마예요?" → "프로"는 영어가 아니라서 못 찾음 ❌
- "팀용 요금제 알려주세요" → "pro"라는 키워드가 없어서 못 찾음 ❌

**해결책**: 의미 기반 검색 (Semantic Search) → 벡터 임베딩 사용!

---

## 3. Part 2: 문서 준비 및 청킹(Chunking)

### 청킹이란?

대용량 문서를 **작은 조각(chunk)** 으로 나누는 과정입니다.

### 왜 청킹이 필요한가?

1. **임베딩 모델 제한**: 한 번에 처리할 수 있는 텍스트 길이 제한
2. **검색 정확도**: 작은 조각이 더 정확한 매칭
3. **컨텍스트 윈도우**: LLM에 전달할 수 있는 토큰 수 제한

### 청킹 전략

| 전략 | 설명 | 장점 | 단점 |
|------|------|------|------|
| 고정 크기 | N개 문자마다 분할 | 간단 | 문맥 무시 |
| 문장 단위 | 문장별로 분할 | 의미 보존 | 크기 불균등 |
| 재귀적 분할 | 계층적으로 분할 | 유연함 | 복잡함 |

### 재귀적 분할
재귀적 분할은 문서의 구조를 최대한 보존하면서 설정한 **Chunk Size(청크 크기)** 에 도달할 때까지 구분자(Separator)의 우선순위에 따라 반복적으로 텍스트를 쪼개는 방식입니다. 단순히 글자 수로만 자르면 문장의 중간이 끊겨 의미가 훼손될 수 있지만, 재귀적 방식은 "문단 → 문장 → 단어" 순으로 최대한 자연스러운 경계선에서 문서를 나눕니다.  

#### 핵심 작동 원리
구분자 리스트 정의: 보통 ["\n\n", "\n", " ", ""] 순서의 리스트를 사용합니다. 단계적 분할: * 먼저 가장 큰 단위인 문단(\n\n)으로 나눕니다. 나눠진 덩어리가 여전히 설정한 Chunk Size보다 크다면, 그다음 구분자인 줄바꿈(\n)으로 다시 나눕니다. 그래도 크다면 공백( ) 단위로, 마지막에는 글자 단위로 내려가며 크기를 맞춥니다.

#### 왜 재귀적 분할을 사용해야 할까?
문맥 보존: 문단이나 문장이 잘리지 않도록 노력하기 때문에, LLM이 정보를 검색했을 때 앞뒤 맥락을 훨씬 더 잘 이해합니다. 유연성: 문서마다 문장의 길이나 문단의 구조가 다르더라도, 정해진 규칙에 따라 유동적으로 대응합니다. 
검색 품질 향상: 의미적으로 완결된 텍스트 뭉치가 벡터 데이터베이스에 저장되므로, 사용자 질문과 유사도를 계산할 때 더 정확한 결과가 나옵니다. 


이어지는 예제에서는 재귀적 분할을 위하여  **RecursiveCharacterTextSplitter**를 사용합니다.


```python
# 전체 문서의 문자 수와 토큰 수 확인
entire_knowledge_base = ""

for content in knowledge.values():
    entire_knowledge_base += content + "\n\n"

print(f"📊 전체 문서 통계:")
print(f"  총 문자 수: {len(entire_knowledge_base):,}")

# 토큰 수 계산
encoding = tiktoken.encoding_for_model(MODEL)
tokens = encoding.encode(entire_knowledge_base)
print(f"  총 토큰 수: {len(tokens):,}")
print(f"\n💡 이 모든 내용을 한 번에 LLM에 전달하면 비용이 많이 듭니다!")
print(f"   청킹을 통해 관련 부분만 찾아서 전달합시다.")
```



<div class="nb-output">

```text
📊 전체 문서 통계:
  총 문자 수: 22,448
  총 토큰 수: 12,452

💡 이 모든 내용을 한 번에 LLM에 전달하면 비용이 많이 듭니다!
   청킹을 통해 관련 부분만 찾아서 전달합시다.
```

</div>



```python
# LangChain의 DirectoryLoader로 문서 로드
folders = glob.glob("sample_docs/*")
documents = []

for folder in folders:
    doc_type = os.path.basename(folder)
    loader = DirectoryLoader(
        folder, 
        glob="**/*.md", 
        loader_cls=TextLoader, 
        loader_kwargs={'encoding': 'utf-8'}
    )
    folder_docs = loader.load()
    
    # 메타데이터 추가
    for doc in folder_docs:
        doc.metadata["doc_type"] = doc_type
        documents.append(doc)

print(f"✅ {len(documents)}개의 문서를 로드했습니다.")
print(f"\n첫 번째 문서 정보:")
print(f"  타입: {documents[0].metadata['doc_type']}")
print(f"  파일: {Path(documents[0].metadata['source']).name}")
print(f"  내용 미리보기:\n{documents[0].page_content[:200]}...")
```



<div class="nb-output">

```text
✅ 9개의 문서를 로드했습니다.

첫 번째 문서 정보:
  타입: products
  파일: cloudstore_pro.md
  내용 미리보기:
# CloudStore Pro

## 개요

CloudStore Pro는 중소기업과 전문가를 위한 고급 클라우드 스토리지 솔루션입니다. 향상된 협업 기능과 관리 도구를 제공합니다.

## 주요 기능

### 스토리지 용량
- 사용자당 1TB 제공
- 팀 공유 스토리지: 추가 5TB
- 필요시 용량 확장 가능

### 고급 협업 기능
- 실시간 문서 공동 편...
```

</div>



```python
# RecursiveCharacterTextSplitter로 청킹 (langchain_text_splitters 사용)
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,        # 각 청크의 최대 크기 (문자 수)
    chunk_overlap=200,      # 청크 간 겹치는 부분 (문맥 유지를 위해)
    length_function=len,
    is_separator_regex=False,
)

chunks = text_splitter.split_documents(documents)

print(f"✂️  청킹 결과:")
print(f"  원본 문서: {len(documents)}개")
print(f"  청크 개수: {len(chunks)}개")
print(f"  평균 청크 크기: {sum(len(c.page_content) for c in chunks) // len(chunks):,} 문자")

print(f"\n첫 번째 청크:")
print(f"{'='*60}")
print(chunks[0].page_content)
print(f"{'='*60}")
print(f"메타데이터: {chunks[0].metadata}")
```



<div class="nb-output">

```text
✂️  청킹 결과:
  원본 문서: 9개
  청크 개수: 29개
  평균 청크 크기: 873 문자

첫 번째 청크:
============================================================
# CloudStore Pro

## 개요

CloudStore Pro는 중소기업과 전문가를 위한 고급 클라우드 스토리지 솔루션입니다. 향상된 협업 기능과 관리 도구를 제공합니다.

## 주요 기능

### 스토리지 용량
- 사용자당 1TB 제공
- 팀 공유 스토리지: 추가 5TB
- 필요시 용량 확장 가능

### 고급 협업 기능
- 실시간 문서 공동 편집
- 댓글 및 피드백 기능
- 버전 관리 (최대 30일)
- 팀 폴더 및 권한 관리

### 관리 도구
- 중앙 관리 콘솔
- 사용자 활동 로그
- 팀 사용량 통계
- 멤버 초대 및 관리

### 통합 기능
- Microsoft Office 통합
- Google Workspace 연동
- Slack, Teams 알림
- Webhook을 통한 커스텀 통합

### 보안 및 규정 준수
- 고급 암호화 옵션
- 감사 로그
- GDPR 준수
- SSO (Single Sign-On) 지원

## 가격

- Pro 플랜: $15/월/사용자
- 최소 3명 이상
- 연간 결제시 20% 할인

... (출력 14줄 생략)
```

</div>


### chunk_size와 chunk_overlap 파라미터

**chunk_size**: 각 청크의 최대 문자 수
- 너무 크면: 검색 정확도 ↓, 비용 ↑
- 너무 작으면: 문맥 손실, 청크 개수 ↑
- 권장: 500-1500 문자

**chunk_overlap**: 청크 간 겹치는 부분
- 문맥이 청크 경계에서 끊기는 것 방지
- 권장: chunk_size의 10-20%

예시:
```
chunk_size=1000, chunk_overlap=200

청크 1: [0----800====1000]
청크 2:       [800====1000----1800====2000]
청크 3:                    [1800====2000----2800]
              ^^^^^ 겹치는 부분 (overlap)
```

---

## 4. Part 3: 벡터 임베딩과 ChromaDB

### ChromaDB란?

벡터를 저장하고 검색하는 데 특화된 오픈소스 데이터베이스입니다.

### 임베딩 모델 선택

| 모델 | 제공자 | 차원 | 비용 | 성능 |
|------|--------|------|------|------|
| text-embedding-3-small | OpenAI | 1536 | 유료 | 높음 |
| text-embedding-3-large | OpenAI | 3072 | 유료 | 매우 높음 |
| all-MiniLM-L6-v2 | HuggingFace | 384 | **무료** | 중간 |

비용을 고려하여 **all-MiniLM-L6-v2** (HuggingFace)를 사용합니다.


```python
# 임베딩 모델 선택
#embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# 또는 OpenAI 임베딩을 사용하려면:
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

print("✅ 임베딩 모델 로드 완료")
print(f"   모델: text-embedding-3-small")
```



<div class="nb-output">

```text
✅ 임베딩 모델 로드 완료
   모델: all-MiniLM-L6-v2 (HuggingFace)
   벡터 차원: 384
```

</div>



```python
# ChromaDB 벡터 스토어 생성
db_name = "cloudstore_vector_db"

# 기존 DB가 있다면 삭제 (처음부터 다시 만들기)
if os.path.exists(db_name):
    Chroma(persist_directory=db_name, embedding_function=embeddings).delete_collection()
    print("🗑️  기존 벡터 DB 삭제")

# 새 벡터 스토어 생성, chunks 를 embedding 파라미터에 전달된 모델로 벡터화함.
print("🔄 벡터 스토어 생성 중...")
vectorstore = Chroma.from_documents(
    documents=chunks, 
    embedding=embeddings, 
    persist_directory=db_name
)

print(f"✅ 벡터 스토어 생성 완료!")
print(f"   저장 위치: {db_name}/")
print(f"   총 벡터 개수: {vectorstore._collection.count():,}")
```



<div class="nb-output">

```text
🗑️  기존 벡터 DB 삭제
🔄 벡터 스토어 생성 중...
✅ 벡터 스토어 생성 완료!
   저장 위치: cloudstore_vector_db/
   총 벡터 개수: 29
```

</div>



```python
# 벡터 정보 확인
collection = vectorstore._collection
count = collection.count()

# 샘플 벡터 가져오기
sample_embedding = collection.get(limit=1, include=["embeddings"])["embeddings"][0]
dimensions = len(sample_embedding)

print(f"📊 벡터 스토어 정보:")
print(f"   총 벡터 개수: {count:,}")
print(f"   벡터 차원: {dimensions:,}")
print(f"\n   샘플 벡터의 첫 10개 값:")
print(f"   {sample_embedding[:10]}")
```



<div class="nb-output">

```text
📊 벡터 스토어 정보:
   총 벡터 개수: 29
   벡터 차원: 1,536

   샘플 벡터의 첫 10개 값:
   [ 0.0128394   0.0292401  -0.00289225 -0.04456824  0.03084372 -0.02102413
 -0.02617863  0.04598442 -0.01276651 -0.02486658]
```

</div>


### 벡터 검색 테스트

이제 벡터 스토어에서 유사한 문서를 검색해봅시다.


```python
# 유사도 검색 테스트
query = "Pro 플랜의 가격은 얼마인가요?"

# top_k개의 가장 유사한 문서 검색
results = vectorstore.similarity_search(query, k=3)

print(f"질문: '{query}'\n")
print(f"{'='*60}")
print(f"가장 유사한 {len(results)}개의 청크:\n")

for i, doc in enumerate(results, 1):
    print(f"[{i}] 출처: {Path(doc.metadata['source']).name}")
    print(f"    타입: {doc.metadata['doc_type']}")
    print(f"    내용:\n{doc.page_content[:3000]}...\n")
    print(f"{'-'*60}\n")
```


### 유사도 점수와 함께 검색


```python
# 유사도 점수와 함께 검색
results_with_scores = vectorstore.similarity_search_with_score(query, k=5)

print(f"질문: '{query}'\n")
print(f"{'='*60}\n")

for i, (doc, score) in enumerate(results_with_scores, 1):
    print(f"[{i}] 유사도 점수: {score:.4f}")
    print(f"    출처: {Path(doc.metadata['source']).name}")
    print(f"    내용 미리보기: {doc.page_content[:150]}...\n")
```



<div class="nb-output">

```text
질문: 'Pro 플랜의 가격은 얼마인가요?'

============================================================

[1] 유사도 점수: 1.0556
    출처: pricing.md
    내용 미리보기: **할인:**
- Pro 플랜: 50% 할인
- Enterprise 플랜: 40% 할인

### 비영리 단체

**자격 요건:**
- 등록된 비영리 단체
- 501(c)(3) 증명서
- 비영리 사명 문서

**할인:**
- Pro 플랜: 40% 할인
- Enterpri...

[2] 유사도 점수: 1.0634
    출처: pricing.md
    내용 미리보기: # CloudStore 요금제 가이드

## 요금제 비교

모든 플랜에는 기본 기능이 포함되어 있으며, 필요에 따라 업그레이드할 수 있습니다.

| 기능 | Basic | Pro | Enterprise |
|------|-------|-----|------------|...

[3] 유사도 점수: 1.1295
    출처: pricing.md
    내용 미리보기: ### Q: 사용자 수를 늘리거나 줄일 수 있나요?

A: 네, Pro와 Enterprise는 언제든지 사용자를 추가/제거할 수 있습니다. 비용은 비례 계산됩니다.

### Q: 무료 평가판이 있나요?

A: Pro와 Enterprise 플랜은 14일 무료 평가판을 제...

[4] 유사도 점수: 1.1670
    출처: pricing.md
    내용 미리보기: **포함 사항:**
- 100GB 스토리지
- 무료 플랜의 모든 기능
- 우선 이메일 지원
- 확장된 공유 옵션

... (출력 28줄 생략)
```

</div>


---

## 5. Part 4: RAG 시스템 구현

이제 벡터 검색과 LLM을 결합하여 완전한 RAG 시스템을 만듭니다.


```python
def rag_query(question, top_k=5):
    """
    RAG를 사용하여 질문에 답변
    
    Args:
        question: 사용자 질문
        top_k: 검색할 문서 개수
    """
    print(f"질문: {question}\n")
    
    # Step 1: 벡터 검색으로 관련 문서 찾기
    print("📚 관련 문서 검색 중...")
    results = vectorstore.similarity_search(question, k=top_k)
    
    print(f"   ✅ {len(results)}개의 관련 문서를 찾았습니다.\n")
    
    # 검색된 문서 표시
    for i, doc in enumerate(results, 1):
        print(f"   [{i}] {Path(doc.metadata['source']).name}")
    
    # Step 2: 컨텍스트 구성
    context = "\n\n".join([doc.page_content for doc in results])
    
    # Step 3: LLM에 질문 + 컨텍스트 전달
    print("\n🤖 AI 답변 생성 중...\n")
    
    system_message = f"""당신은 CloudStore의 문서 검색 어시스턴트입니다.
주어진 문서 정보를 바탕으로 정확하게 답변하세요.
문서에 없는 내용은 '문서에서 해당 정보를 찾을 수 없습니다'라고 답하세요.

관련 문서:
{context}
"""
   
    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": question}
    ]
    
    response = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        temperature=0.3  # 낮은 temperature로 사실 기반 답변 유도
    )
    
    answer = response.choices[0].message.content
    
    print(f"{'='*60}")
    print(f"답변:\n{answer}")
    print(f"{'='*60}")
    
    return answer, results
```


### RAG 시스템 테스트


```python
# 테스트 1: 가격 관련 질문
answer, docs = rag_query("Pro 플랜의 월 요금은 얼마인가요?",4)
```



<div class="nb-output">

```text
질문: Pro 플랜의 월 요금은 얼마인가요?

📚 관련 문서 검색 중...
   ✅ 4개의 관련 문서를 찾았습니다.

   [1] pricing.md
   [2] pricing.md
   [3] pricing.md
   [4] pricing.md

🤖 AI 답변 생성 중...

============================================================
답변:
Pro 플랜의 월 요금은 사용자당 $15입니다.
============================================================
```

</div>



```python
# 테스트 2: API 관련 질문
answer, docs = rag_query("파일을 업로드하는 API는 어떻게 사용하나요?")
```



<div class="nb-output">

```text
질문: 파일을 업로드하는 API는 어떻게 사용하나요?

📚 관련 문서 검색 중...
   ✅ 5개의 관련 문서를 찾았습니다.

   [1] quickstart.md
   [2] storage_api.md
   [3] storage_api.md
   [4] quickstart.md
   [5] storage_api.md

🤖 AI 답변 생성 중...

============================================================
답변:
파일을 업로드하는 API는 다음과 같이 사용합니다:

**POST** `/api/v1/files/upload`

파일을 CloudStore에 업로드합니다.

**Request:**
```json
{
  "file": "<binary_data>",
  "path": "/documents/report.pdf",
  "metadata": {
    "description": "Monthly report",
    "tags": ["report", "monthly"]
  }
}
```

**Response:**
```json
{
  "file_id": "f_1234567890",
  "name": "report.pdf",
  "path": "/documents/report.pdf",
  "size": 2048576,
  "created_at": "2024-01-15T10:30:00Z",
  "download_url": "https://cloudstore.com/download/f_1234567890"
}
```

이 API를 사용하려면 API 키가 필요하며, 이는 요청 헤더에 포함시켜야 합니다:

```
Authorization: Bearer YOUR_API_KEY
```
... (출력 1줄 생략)
```

</div>



```python
# 테스트 3: 보안 관련 질문
answer, docs = rag_query("2단계 인증을 설정하는 방법을 알려주세요")
```



<div class="nb-output">

```text
질문: 2단계 인증을 설정하는 방법을 알려주세요

📚 관련 문서 검색 중...
   ✅ 5개의 관련 문서를 찾았습니다.

   [1] auth_api.md
   [2] security.md
   [3] security.md
   [4] security.md
   [5] auth_api.md

🤖 AI 답변 생성 중...

============================================================
답변:
2단계 인증을 설정하는 방법은 다음과 같습니다:

1. 설정 > 보안으로 이동합니다.
2. "2단계 인증 활성화" 버튼을 클릭합니다.
3. 인증 앱을 선택합니다:
   - Google Authenticator
   - Microsoft Authenticator
   - Authy
4. QR 코드를 스캔합니다.
5. 생성된 6자리 코드를 입력합니다.
6. 복구 코드를 안전한 곳에 저장합니다.
============================================================
```

</div>



```python
# 테스트 4: 제품 비교 질문
answer, docs = rag_query(
    "Basic 플랜과 Pro 플랜의 차이점은 무엇인가요?",
    top_k=5 # 더 많은 문서 검색
)
```



<div class="nb-output">

```text
질문: Basic 플랜과 Pro 플랜의 차이점은 무엇인가요?

📚 관련 문서 검색 중...
   ✅ 5개의 관련 문서를 찾았습니다.

   [1] pricing.md
   [2] pricing.md
   [3] pricing.md
   [4] pricing.md
   [5] pricing.md

🤖 AI 답변 생성 중...

============================================================
답변:
Basic 플랜과 Pro 플랜의 주요 차이점은 다음과 같습니다:

**Basic 플랜:**
- 스토리지: 10GB - 100GB
- 파일 크기 제한: 2GB
- API 호출 제한: 1,000/일
- 버전 기록: 7일
- 공유 링크: 10개
- 암호화: AES-256
- 실시간 협업, 팀 관리, 감사 로그, SSO, 전담 지원 기능이 없음
- SLA: 없음
- 가격: 무료 - $5/월

**Pro 플랜:**
- 스토리지: 1TB/사용자
- 파일 크기 제한: 10GB
- API 호출 제한: 10,000/일
- 버전 기록: 30일
- 공유 링크: 무제한
- 암호화: AES-256
- 실시간 협업, 팀 관리, 감사 로그 기능이 포함되어 있음
- SSO, 전담 지원 기능이 없음
- SLA: 99.9%
- 가격: $15/월/사용자
============================================================
```

</div>


---

## 6. Part 5: 벡터 시각화

벡터가 어떻게 분포되어 있는지 시각화하여 RAG의 작동 원리를 직관적으로 이해해봅시다.

### t-SNE란?

**t-SNE** (t-distributed Stochastic Neighbor Embedding)는 고차원 데이터를 2D 또는 3D로 축소하는 기법입니다.

- 384차원 벡터 → 2D/3D로 변환
- 비슷한 벡터는 가까이, 다른 벡터는 멀리 배치
- 시각화를 통해 클러스터링 확인


```python
# 모든 벡터와 메타데이터 가져오기
result = collection.get(include=['embeddings', 'documents', 'metadatas'])

vectors = np.array(result['embeddings'])
documents = result['documents']
metadatas = result['metadatas']

# 문서 타입 추출
doc_types = [metadata['doc_type'] for metadata in metadatas]

# 문서 타입별 색상 지정
color_map = {
    'products': 'blue',
    'api_docs': 'green',
    'guides': 'red'
}
colors = [color_map.get(t, 'gray') for t in doc_types]

print(f"📊 시각화 준비:")
print(f"   벡터 개수: {len(vectors):,}")
print(f"   벡터 차원: {vectors.shape[1]:,}")
print(f"   문서 타입 분포:")
for doc_type in set(doc_types):
    count = doc_types.count(doc_type)
    print(f"     - {doc_type}: {count}개")
```



<div class="nb-output">

```text
📊 시각화 준비:
   벡터 개수: 29
   벡터 차원: 1,536
   문서 타입 분포:
     - api_docs: 12개
     - guides: 14개
     - products: 3개
```

</div>


### 2D 시각화


```python
# t-SNE로 2D로 축소
print("🔄 t-SNE로 차원 축소 중 (384D → 2D)...")

tsne = TSNE(n_components=2, random_state=42, perplexity=28)
reduced_vectors_2d = tsne.fit_transform(vectors)

print("✅ 차원 축소 완료!")

# 2D 산점도 생성
fig = go.Figure(data=[go.Scatter(
    x=reduced_vectors_2d[:, 0],
    y=reduced_vectors_2d[:, 1],
    mode='markers',
    marker=dict(size=8, color=colors, opacity=0.7),
    text=[f"타입: {t}<br>내용: {d[:100]}..." for t, d in zip(doc_types, documents)],
    hoverinfo='text'
)])

fig.update_layout(
    title='CloudStore 문서 벡터 2D 시각화',
    xaxis_title='차원 1',
    yaxis_title='차원 2',
    width=900,
    height=700,
    margin=dict(r=20, b=10, l=10, t=40)
)

fig.show()

print("\n💡 시각화 해석:")
print("   - 파란색: 제품 문서 (products)")
print("   - 초록색: API 문서 (api_docs)")
print("   - 빨간색: 가이드 (guides)")
print("   - 비슷한 내용의 문서끼리 가까이 위치합니다!")
```



<div class="nb-output">

```text
🔄 t-SNE로 차원 축소 중 (384D → 2D)...
✅ 차원 축소 완료!

💡 시각화 해석:
   - 파란색: 제품 문서 (products)
   - 초록색: API 문서 (api_docs)
   - 빨간색: 가이드 (guides)
   - 비슷한 내용의 문서끼리 가까이 위치합니다!
```

</div>


### 3D 시각화


```python
# t-SNE로 3D로 축소
print("🔄 t-SNE로 차원 축소 중 (384D → 3D)...")

tsne_3d = TSNE(n_components=3, random_state=42, perplexity=28)
reduced_vectors_3d = tsne_3d.fit_transform(vectors)

print("✅ 차원 축소 완료!")

# 3D 산점도 생성
fig_3d = go.Figure(data=[go.Scatter3d(
    x=reduced_vectors_3d[:, 0],
    y=reduced_vectors_3d[:, 1],
    z=reduced_vectors_3d[:, 2],
    mode='markers',
    marker=dict(size=5, color=colors, opacity=0.8),
    text=[f"타입: {t}<br>내용: {d[:100]}..." for t, d in zip(doc_types, documents)],
    hoverinfo='text'
)])

fig_3d.update_layout(
    title='CloudStore 문서 벡터 3D 시각화',
    scene=dict(
        xaxis_title='차원 1',
        yaxis_title='차원 2',
        zaxis_title='차원 3'
    ),
    width=1000,
    height=800,
    margin=dict(r=10, b=10, l=10, t=40)
)

fig_3d.show()

print("\n💡 3D 시각화는 마우스로 회전하여 다양한 각도에서 볼 수 있습니다!")
```



<div class="nb-output">

```text
🔄 t-SNE로 차원 축소 중 (384D → 3D)...
✅ 차원 축소 완료!

💡 3D 시각화는 마우스로 회전하여 다양한 각도에서 볼 수 있습니다!
```

</div>


---

## 7. Part 6: 성능 비교 및 최적화

### 키워드 검색 vs 벡터 검색 비교


```python
# 비교 테스트 케이스
test_questions = [
    "팀용 요금제의 가격은?",
    "API 키를 어떻게 만드나요?",
    "보안을 강화하는 방법",
    "무료로 사용할 수 있나요?"
]

print("🔬 키워드 검색 vs 벡터 검색 비교\n")
print(f"{'='*80}\n")

for question in test_questions:
    print(f"질문: '{question}'\n")
    
    # 키워드 검색
    keyword_results = get_relevant_context_keyword(question)
    print(f"  키워드 검색: {len(keyword_results)}개 문서 발견")
    
    # 벡터 검색
    vector_results = vectorstore.similarity_search(question, k=3)
    print(f"  벡터 검색: {len(vector_results)}개 문서 발견")
    if vector_results:
        print(f"    → {Path(vector_results[0].metadata['source']).name}")
    
    print()
```


### 청크 크기 최적화 팁

#### 실험 결과 (일반적인 가이드라인)

| chunk_size | 검색 정확도 | 처리 속도 | 비용 | 추천 용도 |
|------------|------------|----------|------|----------|
| 200-300 | ⭐⭐ | ⭐⭐⭐ | $ | 짧은 질문 |
| 500-1000 | ⭐⭐⭐ | ⭐⭐ | $$ | 일반적 |
| 1500-2000 | ⭐⭐ | ⭐ | $$$ | 긴 문맥 필요 |

#### 최적화 체크리스트

- [ ] 문서 타입별로 다른 chunk_size 사용
- [ ] chunk_overlap은 chunk_size의 10-20%
- [ ] 검색 후 재순위화(re-ranking) 고려
- [ ] 메타데이터 필터링 활용
- [ ] 임베딩 모델 벤치마크

### 비용 최적화

**임베딩 비용 절감:**
- HuggingFace 모델 사용 (무료)
- 임베딩 캐싱
- 배치 처리

**LLM 비용 절감:**
- top_k 값 최소화 (필요한 만큼만)
- 짧은 청크 사용
- temperature 낮추기 (사실 기반 답변)

---

## 8. 요약

이번 노트북에서 배운 내용:

### 핵심 개념

1. **청킹(Chunking)**: 대용량 문서를 작은 조각으로 분할
   - RecursiveCharacterTextSplitter 사용
   - chunk_size와 chunk_overlap 최적화

2. **벡터 임베딩**: 텍스트를 고차원 벡터로 변환
   - HuggingFace all-MiniLM-L6-v2 모델 사용
   - 384차원 벡터 생성

3. **ChromaDB**: 벡터 데이터베이스로 효율적 검색
   - 유사도 기반 검색
   - 메타데이터 필터링

4. **RAG 파이프라인**: 검색 + 생성 통합
   - 벡터 검색으로 관련 문서 찾기
   - 컨텍스트 주입하여 LLM 호출

5. **시각화**: t-SNE로 벡터 분포 확인
   - 2D/3D 산점도
   - 문서 클러스터링 확인

### 11번 노트북과의 차이

| 항목 | 11번 (기본 RAG) | 12번 (고급 RAG) |
|------|----------------|----------------|
| 검색 방식 | numpy 코사인 유사도 | ChromaDB 벡터 검색 |
| 문서 처리 | 작은 예제 | LangChain 청킹 |
| 확장성 | 소규모 | 대규모 가능 |
| 시각화 | 없음 | t-SNE 시각화 |
| 프로덕션 | 학습용 | 실전 활용 가능 |

### 프로덕션 고려사항

실제 서비스에 적용하려면:

1. **확장성**
   - Pinecone, Weaviate 등 관리형 벡터 DB 사용
   - 분산 처리

2. **성능**
   - 임베딩 캐싱
   - 하이브리드 검색 (키워드 + 벡터)
   - 재순위화(re-ranking)

3. **모니터링**
   - 검색 품질 메트릭
   - 사용자 피드백 수집
   - A/B 테스팅

4. **비용 최적화**
   - 오픈소스 임베딩 모델
   - 배치 처리
   - 캐싱 전략

### 다음 단계

더 깊이 학습하려면:

- **고급 RAG 기법**
  - 하이브리드 검색 (BM25 + 벡터)
  - 재순위화 (Cross-Encoder)
  - 쿼리 확장 (Query Expansion)

- **다른 벡터 DB**
  - Pinecone (관리형)
  - Weaviate (오픈소스)
  - Qdrant (고성능)

- **LangChain 고급 기능**
  - RetrievalQA 체인
  - 대화 메모리
  - 에이전트

---

## 참고 자료

- [LangChain 공식 문서](https://python.langchain.com/docs/)
- [ChromaDB 문서](https://docs.trychroma.com/)
- [HuggingFace Sentence Transformers](https://www.sbert.net/)
- [RAG 논문 (Lewis et al., 2020)](https://arxiv.org/abs/2005.11401)
- [t-SNE 시각화 가이드](https://distill.pub/2016/misread-tsne/)

---

**축하합니다! 🎉**

실전에서 사용 가능한 RAG 시스템을 구축했습니다. 이제 자신만의 문서 검색 AI를 만들어보세요!
