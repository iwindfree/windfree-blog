---
title: "Vector Embeddings와 RAG 기초"
author: iwindfree
pubDatetime: 2025-02-10T09:00:00Z
slug: "llm-vector-embeddings-rag"
category: "AI Engineering"
tags: ["ai", "llm", "rag", "embeddings"]
description: "이번 노트북에서는 현대 AI 시스템의 핵심 기술인 벡터 임베딩Vector Embeddings 과 RAGRetrieval-Augmented Generation 에 대해 알아봅니다."
---

이번 노트북에서는 현대 AI 시스템의 핵심 기술인 **벡터 임베딩(Vector Embeddings)** 과 **RAG(Retrieval-Augmented Generation)** 에 대해 알아봅니다.

## 개요

| 주제 | 내용 |
|------|------|
| 벡터 임베딩 | 텍스트를 고차원 벡터로 변환하는 기술 |
| 유사도 계산 | 벡터 간 의미적 유사성 측정 |
| RAG | 외부 지식을 활용한 LLM 응답 생성 |
| 실전 활용 | 문서 검색과 질의응답 시스템 구축 |

## 학습 목표

1. 벡터 임베딩의 개념과 작동 원리 이해하기
2. OpenAI Embeddings API를 사용하여 텍스트를 벡터로 변환하기
3. 코사인 유사도를 활용한 의미적 유사성 계산하기
4. RAG의 개념과 필요성 이해하기
5. 간단한 RAG 시스템 구현하기

---

## 1. 벡터 임베딩(Vector Embeddings)이란?

### 개념

벡터 임베딩은 텍스트, 이미지, 오디오 등의 데이터를 **고차원 벡터 공간의 점**으로 표현하는 기술입니다. 의미가 비슷한 데이터는 벡터 공간에서 가까운 위치에 배치됩니다.

### 왜 필요한가?

- **의미적 검색**: 키워드가 아닌 의미로 검색 가능
- **유사도 계산**: 두 텍스트의 의미적 유사성을 수치로 측정
- **클러스터링**: 비슷한 내용을 자동으로 그룹화
- **추천 시스템**: 사용자 취향과 유사한 콘텐츠 추천

### 작동 원리

```
"강아지가 공원에서 뛰어놀고 있다" → [0.2, -0.5, 0.8, ..., 0.3]  (1536차원 벡터)
"개가 산책하고 있어요"            → [0.21, -0.48, 0.82, ..., 0.29]
"주식 시장이 하락했다"            → [-0.7, 0.3, -0.1, ..., 0.9]
```

위의 첫 두 문장은 의미가 비슷하므로 벡터 공간에서 가까운 위치에 놓입니다.

## 1.1 임베딩 모델(Embedding Model)의 이해

### 임베딩 모델이란?

임베딩 모델은 **텍스트를 벡터로 변환하는 신경망 모델**입니다. 대부분의 현대 임베딩 모델은 **Transformer 아키텍처** (BERT, RoBERTa 등)를 기반으로 합니다. 개발자 관점에서 임베딩 모델은 텍스트를 ‘의미 좌표’로 바꿔서
검색, 비교, 분류를 가능하게 하는 핵심 요소입니다.

```
텍스트 입력 → [Transformer 인코더] → 고차원 벡터 출력
                    ↓
            문맥을 이해하고
            의미를 압축하여
            수치로 표현
```

### 임베딩 모델의 종류

| 유형 | 설명 | 적합한 용도 | 예시 |
|------|------|-------------|------|
| **대칭형 (Symmetric)** | 입력 쌍이 동등한 형태 | 문장-문장 유사도, 중복 탐지 | sentence-transformers |
| **비대칭형 (Asymmetric)** | 쿼리와 문서가 다른 형태 | 질문-문서 검색, RAG | E5, BGE 시리즈 |

### 임베딩 모델 선택 시 고려사항

1. **차원 수 (Dimensions)**
   - 높은 차원: 더 많은 정보 표현 가능, 저장 공간 증가
   - 낮은 차원: 빠른 검색, 저장 공간 절약
   - 일반적으로 384 ~ 3072 차원 사용

2. **다국어 지원**
   - 영어 전용 모델: 영어 텍스트에서 최고 성능
   - 다국어 모델: 한국어 포함 여러 언어 지원

3. **속도 vs 품질**
   - 작은 모델: 빠르지만 정확도 낮음
   - 큰 모델: 느리지만 높은 품질

4. **비용 (API vs 로컬)**
   - API 모델: 쉬운 사용, 종량제 비용
   - 로컬 모델: 초기 설정 필요, 무료 사용

### 인기 임베딩 모델 비교

| 모델 | 제공자 | 차원 | 다국어 | 특징 |
|------|--------|------|--------|------|
| text-embedding-3-small | OpenAI | 1536 | ✅ | 빠르고 저렴 |
| text-embedding-3-large | OpenAI | 3072 | ✅ | 고품질 |
| all-MiniLM-L6-v2 | HF | 384 | ❌ | 빠름, 무료 |
| multilingual-e5-large | HF | 1024 | ✅ | 다국어 검색에 강함 |
| BAAI/bge-m3 | HF | 1024 | ✅ | 최신 고성능 다국어 |

## 2. 필요한 라이브러리 설치 및 임포트


```python
pip install openai numpy python-dotenv
```



```python
import os
import numpy as np
from openai import OpenAI
from dotenv import load_dotenv

# API 키 로드
load_dotenv(override=True)
api_key = os.getenv('OPENAI_API_KEY')
if not api_key:
    print("❌ API key not found")
else:
    print("✅ API key loaded")
    
client = OpenAI()
```



<div class="nb-output">

```text
✅ API key loaded
```

</div>


## 3. OpenAI Embeddings API 사용하기

### 임베딩 모델

OpenAI는 여러 임베딩 모델을 제공합니다:

| 모델 | 차원 | 성능 | 비용 |
|------|------|------|------|
| text-embedding-3-small | 1536 | 빠르고 저렴 | 낮음 |
| text-embedding-3-large | 3072 | 높은 정확도 | 높음 |

이번 실습에서는 `text-embedding-3-small`을 사용합니다.


```python
# 간단한 텍스트를 벡터로 변환
def get_embedding(text, model="text-embedding-3-small"):
    """텍스트를 벡터로 변환하는 함수"""
    response = client.embeddings.create(
        input=text,
        model=model
    )
    return response.data[0].embedding

# 예시 텍스트
text = "강아지가 공원에서 뛰어놀고 있다"
embedding = get_embedding(text)

print(f"원본 텍스트: {text}")
print(f"임베딩 벡터 차원: {len(embedding)}")
print(f"벡터의 첫 10개 값: {embedding[:10]}")
print(type(embedding))
```



<div class="nb-output">

```text
원본 텍스트: 강아지가 공원에서 뛰어놀고 있다
임베딩 벡터 차원: 1536
벡터의 첫 10개 값: [0.018700942397117615, 0.010911774821579456, 0.0013813195982947946, 0.011796513572335243, 0.024304287508130074, 0.013401186093688011, 0.005126278847455978, 0.007398842368274927, -0.01660185679793358, -0.018145812675356865]
<class 'list'>
```

</div>


### 여러 텍스트 임베딩하기


```python
# 여러 문장 임베딩
sentences = [
    "강아지가 공원에서 뛰어놀고 있다",
    "고양이가 공원에서 뛰어놀고 있다",
    "개가 산책하고 있어요",
    "고양이가 소파에서 자고 있다",
    "주식 시장이 하락했다",
    "경제 뉴스가 발표되었다"
]

embeddings = [get_embedding(sentence) for sentence in sentences]

print(f"총 {len(embeddings)}개의 문장을 임베딩했습니다.")
print(f"각 임베딩 벡터의 차원: {len(embeddings[0])}")
```



<div class="nb-output">

```text
총 6개의 문장을 임베딩했습니다.
각 임베딩 벡터의 차원: 1536
```

</div>


## 3.1 Hugging Face 오픈소스 임베딩 모델

OpenAI API 외에도 **무료 오픈소스 임베딩 모델**을 로컬에서 실행할 수 있습니다. `sentence-transformers` 라이브러리를 사용하면 Hugging Face의 다양한 모델을 쉽게 활용할 수 있습니다.

### 인기 오픈소스 모델

| 모델 | 차원 | 언어 | 특징 |
|------|------|------|------|
| `all-MiniLM-L6-v2` | 384 | 영어 | 빠르고 가벼움, 입문용으로 적합 |
| `paraphrase-multilingual-MiniLM-L12-v2` | 384 | 다국어 | 50+ 언어 지원, 한국어 포함 |
| `BAAI/bge-m3` | 1024 | 다국어 | 최신 고성능 모델, 검색에 강함 |
| `intfloat/multilingual-e5-large` | 1024 | 다국어 | 다국어 검색 벤치마크 상위 |


```python
# sentence-transformers 라이브러리 설치
%pip install sentence-transformers -q
```



```python
from sentence_transformers import SentenceTransformer

# 다국어 지원 모델 로드 (한국어 포함)
hf_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

print(f"모델 로드 완료!")
print(f"임베딩 차원: {hf_model.get_sentence_embedding_dimension()}")
```



<div class="nb-output">

```text
tokenizer_config.json:   0%|          | 0.00/526 [00:00<?, ?B/s]
모델 로드 완료!
임베딩 차원: 384
```

</div>



```python
# Hugging Face 모델로 텍스트 임베딩하기
hf_sentences = [
    "강아지가 공원에서 뛰어놀고 있다",
    "고양이가 공원에서 뛰어놀고 있다",
    "주식 시장이 하락했다"
]

# 여러 문장을 한 번에 임베딩 (배치 처리)
hf_embeddings = hf_model.encode(hf_sentences)

print(f"임베딩 shape: {hf_embeddings.shape}")
print(f"\n첫 번째 문장 벡터 (처음 10개):")
print(hf_embeddings[0][:10])

# 유사도 계산
from sentence_transformers import util

print("\n=== Hugging Face 모델 유사도 ===")
for i in range(len(hf_sentences)):
    for j in range(i+1, len(hf_sentences)):
        sim = util.cos_sim(hf_embeddings[i], hf_embeddings[j]).item()
        print(f"'{hf_sentences[i][:15]}...' vs '{hf_sentences[j][:15]}...': {sim:.4f}")
```



<div class="nb-output">

```text
임베딩 shape: (3, 384)

첫 번째 문장 벡터 (처음 10개):
[ 0.36907932  0.13543867 -0.14175008  0.04047783 -0.16253845 -0.05617588
  0.311258    0.00239587  0.09228859 -0.03149693]

=== Hugging Face 모델 유사도 ===
'강아지가 공원에서 뛰어놀고 ...' vs '고양이가 공원에서 뛰어놀고 ...': 0.6597
'강아지가 공원에서 뛰어놀고 ...' vs '주식 시장이 하락했다...': 0.3356
'고양이가 공원에서 뛰어놀고 ...' vs '주식 시장이 하락했다...': 0.2879
```

</div>


### OpenAI vs Hugging Face 임베딩 모델 비교

| 항목 | OpenAI API | Hugging Face (로컬) |
|------|------------|---------------------|
| **비용** | 사용량에 따라 과금 | 무료 (컴퓨팅 비용만) |
| **속도** | 네트워크 지연 있음 | 로컬 실행으로 빠름 |
| **프라이버시** | 데이터가 서버로 전송됨 | 데이터가 로컬에 유지 |
| **오프라인** | 인터넷 필요 | 오프라인 사용 가능 |
| **품질** | 일반적으로 높은 품질 | 모델에 따라 다양 |
| **설정** | API 키만 필요 | 모델 다운로드 필요 |
| **확장성** | 무제한 확장 가능 | 하드웨어 제약 |

**언제 무엇을 선택할까요?**
- **OpenAI**: 프로덕션 서비스, 높은 품질이 필요할 때, 인프라 관리 부담을 줄이고 싶을 때
- **Hugging Face**: 비용 절감, 데이터 프라이버시가 중요할 때, 오프라인 환경, 커스터마이징이 필요할 때

## 4. 코사인 유사도(Cosine Similarity)

### 개념

코사인 유사도는 두 벡터 사이의 각도를 이용해 유사성을 측정합니다.

- **1에 가까울수록**: 매우 유사함 (같은 방향)
- **0에 가까울수록**: 무관함 (직각)
- **-1에 가까울수록**: 반대됨 (정반대 방향)

### 계산 공식

```
cosine_similarity = (A · B) / (||A|| × ||B||)
```


```python
def cosine_similarity(vec1, vec2):
    """두 벡터 간의 코사인 유사도 계산"""
    vec1 = np.array(vec1)
    vec2 = np.array(vec2)
    
    dot_product = np.dot(vec1, vec2)
    norm_vec1 = np.linalg.norm(vec1)
    norm_vec2 = np.linalg.norm(vec2)
    
    return dot_product / (norm_vec1 * norm_vec2)

# 테스트
similarity = cosine_similarity(embeddings[0], embeddings[1])
print(f"'{sentences[0]}'")
print(f"'{sentences[1]}'")
print(f"유사도: {similarity:.4f}")
```



<div class="nb-output">

```text
'강아지가 공원에서 뛰어놀고 있다'
'고양이가 공원에서 뛰어놀고 있다'
유사도: 0.6047
```

</div>


### 모든 문장 간 유사도 계산


```python
# 유사도 매트릭스 생성
print("\n=== 문장 간 유사도 매트릭스 ===")
print("\n" + " " * 30, end="")
for i, _ in enumerate(sentences):
    print(f"문장{i+1:2d}", end="  ")
print()

for i, sent1 in enumerate(sentences):
    print(f"문장{i+1} ({sent1[:12]}...)", end=" ")
    for j, sent2 in enumerate(sentences):
        sim = cosine_similarity(embeddings[i], embeddings[j])
        print(f"{sim:6.3f}", end="  ")
    print()
```



<div class="nb-output">

```text

=== 문장 간 유사도 매트릭스 ===

                              문장 1  문장 2  문장 3  문장 4  문장 5  문장 6  
문장1 (강아지가 공원에서 뛰어...)  1.000   0.605   0.158   0.339   0.085   0.097  
문장2 (고양이가 공원에서 뛰어...)  0.605   1.000   0.164   0.709   0.150   0.085  
문장3 (개가 산책하고 있어요...)  0.158   0.164   1.000   0.162   0.159   0.168  
문장4 (고양이가 소파에서 자고...)  0.339   0.709   0.162   1.000   0.139   0.105  
문장5 (주식 시장이 하락했다...)  0.085   0.150   0.159   0.139   1.000   0.348  
문장6 (경제 뉴스가 발표되었다...)  0.097   0.085   0.168   0.105   0.348   1.000
```

</div>


### 가장 유사한 문장 찾기


```python
def find_most_similar(query, sentences, embeddings, top_k=3):
    """쿼리와 가장 유사한 문장 찾기"""
    query_embedding = get_embedding(query)
    
    similarities = []
    for i, embedding in enumerate(embeddings):
        sim = cosine_similarity(query_embedding, embedding)
        similarities.append((sentences[i], sim))
    
    # 유사도 순으로 정렬
    similarities.sort(key=lambda x: x[1], reverse=True)
    
    return similarities[:top_k]

# 테스트
query = "애완동물이 놀고 있어요"
results = find_most_similar(query, sentences, embeddings)

print(f"질문: '{query}'\n")
print("가장 유사한 문장들:")
for i, (sentence, similarity) in enumerate(results, 1):
    print(f"{i}. {sentence}")
    print(f"   유사도: {similarity:.4f}\n")
```



<div class="nb-output">

```text
질문: '애완동물이 놀고 있어요'

가장 유사한 문장들:
1. 고양이가 소파에서 자고 있다
   유사도: 0.3648

2. 강아지가 공원에서 뛰어놀고 있다
   유사도: 0.3522

3. 고양이가 공원에서 뛰어놀고 있다
   유사도: 0.3508
```

</div>


## 5. RAG (Retrieval-Augmented Generation)란?

### 개념

RAG는 외부 지식을 검색하여 LLM의 응답에 활용하는 기술입니다.

### 왜 필요한가?

LLM의 한계:
- **지식 차단**: 학습 데이터의 시점까지만 알고 있음
- **환각(Hallucination)**: 모르는 내용을 그럴듯하게 지어냄
- **도메인 지식 부족**: 특정 회사나 제품의 최신 정보 모름

RAG의 해결책:
- 실시간으로 최신 정보를 검색하여 제공
- 신뢰할 수 있는 출처 기반 답변
- 도메인 특화 지식베이스 활용

### RAG 파이프라인

```
1. 문서 준비
   └→ 문서들을 임베딩하여 벡터 DB에 저장
   
2. 질문 받기
   └→ 사용자 질문을 임베딩
   
3. 관련 문서 검색
   └→ 유사도가 높은 문서들 찾기
   
4. 컨텍스트 주입
   └→ 검색된 문서와 질문을 함께 LLM에 전달
   
5. 답변 생성
   └→ LLM이 문서를 참고하여 답변
```

## 6. 간단한 RAG 시스템 구현

벡터 DB 없이 numpy만으로 간단한 RAG를 구현해봅시다.


```python
# 1. 지식베이스 준비 (예: 회사 정책 문서)
knowledge_base = [
    "우리 회사의 연차 휴가는 입사 1년 후부터 연 15일이 제공됩니다.",
    "재택근무는 주 2회까지 가능하며, 사전에 팀장의 승인을 받아야 합니다.",
    "점심시간은 12시부터 1시까지이며, 구내식당을 무료로 이용할 수 있습니다.",
    "회사 건물은 오전 8시에 개방되고 오후 10시에 폐쇄됩니다.",
    "신입사원 교육은 입사 첫 주에 3일간 진행되며, 필수 참석입니다.",
    "경조사 휴가는 경조사 종류에 따라 1일에서 5일까지 제공됩니다.",
    "복지포인트는 매년 100만원이 지급되며, 자유롭게 사용할 수 있습니다."
]

print("지식베이스 문서 수:", len(knowledge_base))
print("\n문서 목록:")
for i, doc in enumerate(knowledge_base, 1):
    print(f"{i}. {doc}")
```



<div class="nb-output">

```text
지식베이스 문서 수: 7

문서 목록:
1. 우리 회사의 연차 휴가는 입사 1년 후부터 연 15일이 제공됩니다.
2. 재택근무는 주 2회까지 가능하며, 사전에 팀장의 승인을 받아야 합니다.
3. 점심시간은 12시부터 1시까지이며, 구내식당을 무료로 이용할 수 있습니다.
4. 회사 건물은 오전 8시에 개방되고 오후 10시에 폐쇄됩니다.
5. 신입사원 교육은 입사 첫 주에 3일간 진행되며, 필수 참석입니다.
6. 경조사 휴가는 경조사 종류에 따라 1일에서 5일까지 제공됩니다.
7. 복지포인트는 매년 100만원이 지급되며, 자유롭게 사용할 수 있습니다.
```

</div>



```python
# 2. 모든 문서를 임베딩
print("문서들을 임베딩하는 중...")
kb_embeddings = [get_embedding(doc) for doc in knowledge_base]
print(f"✅ {len(kb_embeddings)}개 문서 임베딩 완료")
```



<div class="nb-output">

```text
문서들을 임베딩하는 중...
✅ 7개 문서 임베딩 완료
```

</div>



```python
# 3. RAG 함수 구현
def rag_query(question, knowledge_base, kb_embeddings, top_k=2):
    """
    RAG를 사용하여 질문에 답변
    
    Args:
        question: 사용자 질문
        knowledge_base: 문서 리스트
        kb_embeddings: 문서 임베딩 리스트
        top_k: 검색할 문서 개수
    """
    # Step 1: 질문 임베딩, 질문을 벡터로 변환
    print(f"질문: {question}\n")
    question_embedding = get_embedding(question)
    
    # Step 2: 유사한 문서 검색
    print("📚 관련 문서 검색 중...")
    similarities = []
    for i, doc_embedding in enumerate(kb_embeddings):
        sim = cosine_similarity(question_embedding, doc_embedding)
        similarities.append((i, knowledge_base[i], sim))
    
    # 유사도 순 정렬
    similarities.sort(key=lambda x: x[2], reverse=True)
    top_docs = similarities[:top_k]
    
    print(f"\n가장 관련있는 {top_k}개 문서:")
    for i, (idx, doc, sim) in enumerate(top_docs, 1):
        print(f"  {i}. (유사도: {sim:.4f}) {doc}")
    
    # Step 3: 컨텍스트 구성
    context = "\n".join([doc for _, doc, _ in top_docs])
    
    # Step 4: LLM에 컨텍스트와 질문 전달
    print("\n🤖 LLM 응답 생성 중...\n")
    messages = [
        {
            "role": "system",
            "content": "당신은 회사 정책에 대해 정확하게 답변하는 HR 어시스턴트입니다. 주어진 문서 정보만을 바탕으로 답변하세요."
        },
        {
            "role": "user",
            "content": f"""다음은 관련 문서입니다:

            {context}

            질문: {question}

            위 문서를 참고하여 질문에 답변해주세요."""
        }
    ]
    
    response = client.chat.completions.create(
        model="gpt-4",
        messages=messages,
        temperature=0.3
    )
    
    answer = response.choices[0].message.content
    
    print("="*60)
    print("답변:")
    print(answer)
    print("="*60)
    
    return answer, top_docs
```


### RAG 시스템 테스트


```python
# 테스트 1: 연차 관련 질문
answer, docs = rag_query(
    "입사하면 휴가를 몇 일이나 쓸 수 있나요?",
    knowledge_base,
    kb_embeddings
)
```



<div class="nb-output">

```text
질문: 입사하면 휴가를 몇 일이나 쓸 수 있나요?

📚 관련 문서 검색 중...

가장 관련있는 2개 문서:
  1. (유사도: 0.5321) 경조사 휴가는 경조사 종류에 따라 1일에서 5일까지 제공됩니다.
  2. (유사도: 0.5179) 우리 회사의 연차 휴가는 입사 1년 후부터 연 15일이 제공됩니다.

🤖 LLM 응답 생성 중...

============================================================
답변:
입사 1년 후부터 연차 휴가 15일이 제공됩니다. 추가로 경조사에 따라 1일에서 5일까지의 휴가를 더 사용할 수 있습니다.
============================================================
```

</div>



```python
# 테스트 2: 재택근무 관련 질문
answer, docs = rag_query(
    "집에서 일하고 싶은데 가능한가요?",
    knowledge_base,
    kb_embeddings
)
```



<div class="nb-output">

```text
질문: 집에서 일하고 싶은데 가능한가요?

📚 관련 문서 검색 중...

가장 관련있는 2개 문서:
  1. (유사도: 0.2739) 재택근무는 주 2회까지 가능하며, 사전에 팀장의 승인을 받아야 합니다.
  2. (유사도: 0.2501) 점심시간은 12시부터 1시까지이며, 구내식당을 무료로 이용할 수 있습니다.

🤖 LLM 응답 생성 중...

============================================================
답변:
네, 가능합니다. 하지만 재택근무는 주 2회까지만 가능하며, 사전에 팀장의 승인을 받아야 합니다.
============================================================
```

</div>



```python
# 테스트 3: 복합 질문
answer, docs = rag_query(
    "신입사원이 알아야 할 중요한 정보는 무엇인가요?",
    knowledge_base,
    kb_embeddings,
    top_k=3
)
```



<div class="nb-output">

```text
질문: 신입사원이 알아야 할 중요한 정보는 무엇인가요?

📚 관련 문서 검색 중...

가장 관련있는 3개 문서:
  1. (유사도: 0.5917) 신입사원 교육은 입사 첫 주에 3일간 진행되며, 필수 참석입니다.
  2. (유사도: 0.2267) 재택근무는 주 2회까지 가능하며, 사전에 팀장의 승인을 받아야 합니다.
  3. (유사도: 0.2228) 우리 회사의 연차 휴가는 입사 1년 후부터 연 15일이 제공됩니다.

🤖 LLM 응답 생성 중...

============================================================
답변:
신입사원이 알아야 할 중요한 정보는 다음과 같습니다. 첫째, 입사 첫 주에 3일간 신입사원 교육이 진행되며, 이는 필수로 참석해야 합니다. 둘째, 재택근무는 주 2회까지 가능하지만, 이를 위해서는 사전에 팀장의 승인을 받아야 합니다. 셋째, 연차 휴가는 입사 1년 후부터 연 15일이 제공됩니다.
============================================================
```

</div>


## 7. RAG vs 일반 LLM 비교

지식베이스에 없는 정보를 물어보면 어떻게 될까요?


```python
# RAG 없이 직접 질문
def ask_without_rag(question):
    """RAG 없이 LLM에 직접 질문"""
    messages = [
        {
            "role": "system",
            "content": "당신은 회사 정책에 대해 답변하는 HR 어시스턴트입니다."
        },
        {
            "role": "user",
            "content": question
        }
    ]
    
    response = client.chat.completions.create(
        model="gpt-4",
        messages=messages,
        temperature=0.3
    )
    
    return response.choices[0].message.content

# 비교 테스트
question = "우리 회사 연차는 며칠인가요?"

print("🔍 RAG 사용 (지식베이스 참고):")
print("="*60)
rag_answer, _ = rag_query(question, knowledge_base, kb_embeddings)

print("\n\n❌ RAG 미사용 (LLM 지식만 사용):")
print("="*60)
no_rag_answer = ask_without_rag(question)
print(no_rag_answer)
print("="*60)

print("\n💡 차이점:")
print("- RAG: 정확한 회사 정책(15일)을 제공")
print("- No RAG: 일반적인 답변이거나 정확하지 않을 수 있음")
```



<div class="nb-output">

```text
🔍 RAG 사용 (지식베이스 참고):
============================================================
질문: 우리 회사 연차는 며칠인가요?

📚 관련 문서 검색 중...

가장 관련있는 2개 문서:
  1. (유사도: 0.5281) 우리 회사의 연차 휴가는 입사 1년 후부터 연 15일이 제공됩니다.
  2. (유사도: 0.2932) 회사 건물은 오전 8시에 개방되고 오후 10시에 폐쇄됩니다.

🤖 LLM 응답 생성 중...

============================================================
답변:
우리 회사의 연차는 연 15일입니다.
============================================================


❌ RAG 미사용 (LLM 지식만 사용):
============================================================
회사의 정책에 따라 다르지만, 일반적으로 한 해에 15일의 연차가 주어지는 것이 표준입니다. 하지만, 이는 근속 연수, 회사의 정책, 그리고 국가의 노동법에 따라 다를 수 있습니다. 정확한 정보는 당사의 인사 담당자에게 문의하시거나, 직원 핸드북을 참조해 주시기 바랍니다.
============================================================

💡 차이점:
- RAG: 정확한 회사 정책(15일)을 제공
- No RAG: 일반적인 답변이거나 정확하지 않을 수 있음
```

</div>


## 8. 실전 팁

### 성능 향상 방법

1. **청크 크기 조정**: 문서를 적절한 크기로 분할
2. **하이브리드 검색**: 키워드 + 벡터 검색 병행
3. **재순위화(Re-ranking)**: 검색 결과를 다시 정렬
4. **메타데이터 활용**: 날짜, 출처 등 추가 정보 활용

### 벡터 데이터베이스

실전에서는 numpy 대신 전문 벡터 DB를 사용합니다:

| 벡터 DB | 특징 | 추천 용도 |
|---------|------|----------|
| **Pinecone** | 완전 관리형, 확장성 | 프로덕션 서비스 |
| **ChromaDB** | 오픈소스, 간단 | 프로토타입, 소규모 |
| **Weaviate** | 오픈소스, 풍부한 기능 | 복잡한 검색 |
| **Qdrant** | 오픈소스, 빠른 성능 | 대용량 데이터 |

### 비용 최적화

- 임베딩 캐싱: 동일한 텍스트는 재사용
- 배치 처리: 여러 텍스트를 한 번에 임베딩
- 작은 모델 사용: `text-embedding-3-small` 선택

## 9. 요약

이번 노트북에서 다룬 내용:

### 핵심 포인트

1. **벡터 임베딩**: 텍스트를 고차원 벡터로 변환하여 의미를 수치화
2. **코사인 유사도**: 두 벡터의 의미적 유사성을 -1~1 범위로 측정
3. **RAG**: 외부 지식을 검색하여 LLM 응답의 정확성 향상
4. **RAG 파이프라인**: 임베딩 → 검색 → 컨텍스트 주입 → 생성
5. **실전 활용**: 문서 검색, 질의응답, 추천 시스템 등

### RAG의 장점

- ✅ 최신 정보 활용 가능
- ✅ 도메인 특화 지식 제공
- ✅ 환각(Hallucination) 감소
- ✅ 출처 추적 가능

### 다음 단계

다음 학습에서는:
- ChromaDB, Pinecone 등 벡터 DB 활용
- 대용량 문서 처리 (청킹 전략)
- 고급 RAG 기법 (하이브리드 검색, 재순위화)
- LangChain을 활용한 RAG 파이프라인

---

**참고 자료**
- [OpenAI Embeddings Guide](https://platform.openai.com/docs/guides/embeddings)
- [Vector Database Comparison](https://github.com/erikbern/ann-benchmarks)
