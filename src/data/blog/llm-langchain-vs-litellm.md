---
title: "LangChain vs LiteLLM 비교 가이드"
author: iwindfree
pubDatetime: 2025-02-11T09:00:00Z
slug: "llm-langchain-vs-litellm"
category: "LLM Engineering"
series: "LLM Engineering"
seriesOrder: 16
tags: ["ai", "llm", "rag", "langchain"]
description: "> 원문: LangChain vs LiteLLM by Hey Amithttps://medium.com/@heyamit10/langchain-vs-litellm-a9b784a2ad1a"
---

> 원문: [LangChain vs LiteLLM by Hey Amit](https://medium.com/@heyamit10/langchain-vs-litellm-a9b784a2ad1a)

## 개요

LangChain과 LiteLLM은 모두 AI 커뮤니티에서 인기 있는 프레임워크이지만, 서로 매우 다른 요구사항을 충족합니다. 이 노트북에서는 두 도구의 특성과 적합한 사용 시나리오를 알아보고, 실제 코드 예제를 통해 비교해봅니다.

| 주제 | 내용 |
|------|------|
| LangChain | 복잡한 워크플로우를 위한 범용 프레임워크 |
| LiteLLM | 경량화된 통합 LLM 인터페이스 |
| 비교 분석 | 장단점, 사용 사례, 선택 가이드 |
| 실습 | 두 프레임워크의 실제 사용 예제 |

## 학습 목표

1. LangChain의 핵심 개념과 사용법 이해하기
2. LiteLLM의 특징과 장점 파악하기
3. 프로젝트 요구사항에 맞는 프레임워크 선택하기
4. 두 프레임워크를 함께 사용하는 방법 익히기

---

## 0. 필요한 라이브러리 설치 및 임포트


```python
pip install openai langchain langchain-openai langchain-community litellm python-dotenv
```



```python
import os
from dotenv import load_dotenv

# API 키 설정
load_dotenv(override=True)
openai_api_key = os.getenv('OPENAI_API_KEY')

if openai_api_key:
    print(f"✅ OpenAI API Key loaded (시작: {openai_api_key[:8]}...)")
else:
    print("❌ OpenAI API Key not found")
```



<div class="nb-output">

```text
✅ OpenAI API Key loaded (시작: sk-proj-...)
```

</div>


---

## 1. LangChain 소개

### 핵심 철학

> "복잡한 문제를 해결하려면 복잡한 도구가 필요하다"

LangChain은 복잡한 워크플로우를 위한 **강력한 범용 프레임워크**입니다. 2022년 Harrison Chase가 시작한 이 프로젝트는 LLM과 외부 데이터 소스, 도구들을 쉽게 연결할 수 있게 해줍니다.

### 주요 특징

| 특징 | 설명 |
|------|------|
| **모듈성 (Modularity)** | 다양한 LLM 작업을 체인으로 연결하여 복잡한 워크플로우 구축 |
| **확장성 (Extensibility)** | 여러 데이터 소스에서 데이터를 가져와 단계별로 처리 가능 |
| **체이닝 컴포넌트** | 다양한 컴포넌트를 연결하여 고급 사용 사례 구현 |
| **외부 도구 통합** | 다양한 외부 도구 및 서비스와 쉽게 통합 |
| **커스터마이징** | 체인의 각 컴포넌트를 세밀하게 제어 가능 |

### 핵심 구성 요소

| 구성 요소 | 설명 |
|----------|------|
| **Models** | OpenAI, Anthropic, HuggingFace 등 다양한 LLM 통합 |
| **Prompts** | 프롬프트 템플릿 및 관리 |
| **Chains** | 여러 컴포넌트를 연결한 워크플로우 |
| **Retrievers** | 벡터 DB, 검색 엔진 등에서 문서 검색 |
| **Memory** | 대화 히스토리 관리 |
| **Agents** | LLM이 도구를 선택하고 실행하는 자율 시스템 |

### 적합한 사용 사례

- 문서 처리 파이프라인
- 자율 에이전트 시스템
- 멀티스텝 추론 시스템
- RAG (Retrieval-Augmented Generation) 파이프라인
- 메모리 통합 (Redis, Vector Store 등)
- 멀티 에이전트 오케스트레이션

### LangChain 기본 사용 예제


```python
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# 1. 모델 초기화
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)

# 2. 프롬프트 템플릿 정의
prompt = ChatPromptTemplate.from_messages([
    ("system", "당신은 {topic} 전문가입니다. 간결하고 명확하게 답변해주세요."),
    ("user", "{question}")
])

# 3. 체인 구성 (LCEL - LangChain Expression Language)
chain = prompt | llm | StrOutputParser()

# 4. 체인 실행
response = chain.invoke({
    "topic": "Python 프로그래밍",
    "question": "리스트 컴프리헨션이란 무엇인가요?"
})

print("🔗 LangChain 응답:")
print(response)
```



<div class="nb-output">

```text
🔗 LangChain 응답:
리스트 컴프리헨션(List Comprehension)은 파이썬에서 리스트를 간결하게 생성할 수 있는 방법입니다. 일반적인 리스트 생성 방식보다 더 짧고 읽기 쉽게 표현할 수 있습니다. 

기본 문법은 다음과 같습니다:

```python
[표현식 for 아이템 in iterable if 조건]
```

예를 들어, 0부터 9까지의 숫자 중 짝수만 포함하는 리스트를 생성하고 싶다면:

```python
even_numbers = [x for x in range(10) if x % 2 == 0]
```

위 코드는 `[0, 2, 4, 6, 8]`을 생성합니다. 리스트 컴프리헨션은 가독성이 좋고, 코드의 간결성을 향상시킵니다.
```

</div>


### LangChain 체이닝 예제

여러 단계를 연결하여 복잡한 워크플로우를 구성할 수 있습니다.


```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

# 리뷰할 코드 (의도적으로 개선이 필요한 코드)
sample_code = """
def calculate_average(numbers):
    total = 0
    for i in range(len(numbers)):
        total = total + numbers[i]
    average = total / len(numbers)
    return average
"""

# 1단계: 코드 문제점 분석
review_prompt = ChatPromptTemplate.from_template(
    "다음 Python 코드의 문제점과 개선할 점을 분석하세요:\n```python\n{code}\n```\n\n문제점:"
)

# 2단계: 개선된 코드 생성
improve_prompt = ChatPromptTemplate.from_template(
    "원본 코드:\n```python\n{code}\n```\n\n발견된 문제점:\n{issues}\n\n위 문제점을 해결한 개선된 코드를 작성하세요:"
)

# 체인 구성: 문제점 분석 → 개선된 코드 생성
review_chain = review_prompt | llm | StrOutputParser()

full_chain = (
    {"issues": review_chain, "code": RunnablePassthrough()}
    | improve_prompt
    | llm
    | StrOutputParser()
)

# 실행
print("📝 원본 코드:")
print(sample_code)
print("\n" + "="*50 + "\n")

result = full_chain.invoke(sample_code)

print("🔧 코드 리뷰 결과 (개선된 코드):")
print(result)
```



<div class="nb-output">

```text
📝 원본 코드:

def calculate_average(numbers):
    total = 0
    for i in range(len(numbers)):
        total = total + numbers[i]
    average = total / len(numbers)
    return average


==================================================

🔧 코드 리뷰 결과 (개선된 코드):
아래는 주어진 문제점을 해결하고 개선된 코드를 작성한 예시입니다. 이 코드는 입력 검증, 빈 리스트 처리, 가독성을 높이기 위해 Python 내장 함수를 사용합니다.

```python
def calculate_average(numbers):
    # 입력 검증: numbers가 리스트인지 확인하고 빈 리스트인지 체크
    if not isinstance(numbers, list) or not numbers:
        return None  # 또는 적절한 예외 처리

    # 모든 요소가 숫자인지 확인
    for number in numbers:
        if not isinstance(number, (int, float)):
            raise ValueError("All elements must be numbers")

    # 총합 계산 및 평균 반환
    total = sum(numbers)
    average = total / len(numbers)
    return average
```

### 개선된 코드의 특징 및 장점
1. **입력 검증**: `numbers`가 리스트인지 확인하고, 빈 리스트인 경우 `None`을 반환하여 안정성을 높였습니다.
2. **모든 요소 타입 검사**: 리스트의 모든 요소가 숫자인지 확인하여, 잘못된 입력에 대해 `ValueError`를 발생시킵니다.
3. **가독성 향상**: `sum()` 함수를 사용하여 코드를 간결하게 만들었습니다. 반복문을 통한 수동 합계 계산을 피했습니다.
4. **Pythonic한 접근**: Python의 내장 기능을 사용하여 코드의 가독성과 효율성을 높였습니다.

이 개선된 코드는 다양한 입력 상황에서 안정적으로 작동하며, 오류 상황을 명확하게 처리합니다.
```

</div>


#### 위 코드 상세 설명

##### LCEL (LangChain Expression Language)

LangChain 0.1 버전부터 도입된 **LCEL**은 체인을 구성하는 선언적 방식입니다. 파이프 연산자(`|`)를 사용하여 컴포넌트들을 직관적으로 연결합니다.

```python
chain = prompt | llm | output_parser
```

##### 핵심 개념

| 개념 | 설명 |
|------|------|
| **Runnable** | LCEL의 기본 단위. 모든 컴포넌트(prompt, llm, parser 등)는 Runnable 인터페이스를 구현 |
| **파이프 연산자 (`\|`)** | 왼쪽 Runnable의 출력을 오른쪽 Runnable의 입력으로 전달 |
| **RunnablePassthrough** | 입력을 그대로 다음 단계로 전달. 여러 입력을 병렬로 처리할 때 유용 |
| **RunnableParallel** | 여러 Runnable을 병렬로 실행하고 결과를 딕셔너리로 결합 |

---

##### 예제 시나리오: 코드 리뷰 체인

이 예제에서는 **문제가 있는 코드**를 입력받아 **문제점 분석 → 개선된 코드 생성**의 2단계를 거칩니다.

**입력 코드의 문제점:**
```python
def calculate_average(numbers):
    total = 0
    for i in range(len(numbers)):      # ❌ 파이썬답지 않은 반복문
        total = total + numbers[i]      # ❌ += 연산자 미사용
    average = total / len(numbers)      # ❌ 빈 리스트 처리 없음 (ZeroDivisionError)
    return average
```

---

##### Step 1: 프롬프트 템플릿 정의

```python
# 1단계: 코드 문제점 분석 프롬프트
review_prompt = ChatPromptTemplate.from_template(
    "다음 Python 코드의 문제점과 개선할 점을 분석하세요:\n```python\n{code}\n```\n\n문제점:"
)

# 2단계: 개선된 코드 생성 프롬프트
improve_prompt = ChatPromptTemplate.from_template(
    "원본 코드:\n```python\n{code}\n```\n\n발견된 문제점:\n{issues}\n\n위 문제점을 해결한 개선된 코드를 작성하세요:"
)
```

- `{code}`: 리뷰할 원본 코드가 들어갈 자리
- `{issues}`: 1단계에서 분석한 문제점이 들어갈 자리

---

##### Step 2: 첫 번째 체인 구성 (문제점 분석)

```python
review_chain = review_prompt | llm | StrOutputParser()
```

이 체인의 동작:

```
입력: "def calculate_average(numbers):..."
  │
  ▼ review_prompt
"다음 Python 코드의 문제점과 개선할 점을 분석하세요:
```python
def calculate_average(numbers):
    total = 0
    ...
```

문제점:"
  │
  ▼ llm
AIMessage(content="1. for i in range(len())은 파이썬답지 않습니다...")
  │
  ▼ StrOutputParser()
"1. for i in range(len())은 파이썬답지 않습니다.
 2. 빈 리스트 입력 시 ZeroDivisionError 발생
 3. total = total + 대신 += 사용 권장"
```

---

##### Step 3: 전체 체인 구성 (핵심!)

```python
full_chain = (
    {"issues": review_chain, "code": RunnablePassthrough()}
    | improve_prompt
    | llm
    | StrOutputParser()
)
```

**`{"issues": review_chain, "code": RunnablePassthrough()}`**

이것이 핵심입니다! 딕셔너리는 **RunnableParallel**의 축약 문법으로, 두 작업을 **병렬로** 실행합니다:

| 키 | 값 | 동작 |
|---|---|------|
| `issues` | `review_chain` | 코드를 분석하여 문제점 추출 |
| `code` | `RunnablePassthrough()` | 원본 코드를 그대로 전달 |

```
입력: "def calculate_average(numbers):..."
                    │
        ┌───────────┴───────────┐
        ▼                       ▼
   review_chain           RunnablePassthrough()
   (문제점 분석)              (원본 코드 유지)
        │                       │
        ▼                       ▼
"1. for i in range..."    "def calculate_average..."
        │                       │
        └───────────┬───────────┘
                    ▼
    {
      "issues": "1. for i in range(len())은 파이썬답지 않습니다...",
      "code": "def calculate_average(numbers):..."
    }
```

**왜 `RunnablePassthrough()`가 필요한가?**

2단계 프롬프트(`improve_prompt`)에는 **원본 코드**와 **문제점** 둘 다 필요합니다. 
- `issues`: 1단계 분석 결과 (새로 생성)
- `code`: 원본 코드 (그대로 유지) ← **RunnablePassthrough()의 역할**

---

##### Step 4: 최종 프롬프트 생성 및 실행

딕셔너리가 `improve_prompt`로 전달되면:

```
"원본 코드:
```python
def calculate_average(numbers):
    total = 0
    for i in range(len(numbers)):
        total = total + numbers[i]
    average = total / len(numbers)
    return average
```

발견된 문제점:
1. for i in range(len())은 파이썬답지 않습니다.
2. 빈 리스트 입력 시 ZeroDivisionError 발생
3. total = total + 대신 += 사용 권장

위 문제점을 해결한 개선된 코드를 작성하세요:"
```

이 프롬프트가 LLM에 전달되어 개선된 코드가 생성됩니다.

---

##### 전체 실행 흐름 요약

```
"def calculate_average(numbers):..."
                │
                ▼
┌─────────────────────────────────────────┐
│  RunnableParallel (병렬 실행)            │
│  ┌───────────────┬─────────────────┐    │
│  │ issues:       │ code:           │    │
│  │ review_chain  │ RunnablePass    │    │
│  │ (문제점 분석)  │ through()       │    │
│  │      │        │ (원본 유지)      │    │
│  │      ▼        │      ▼          │    │
│  │ "1. for i..." │  원본 코드       │    │
│  └───────────────┴─────────────────┘    │
└─────────────────────────────────────────┘
                │
                ▼
    {"issues": "1. for i...", "code": "def calculate..."}
                │
                ▼
┌─────────────────────────────────────────┐
│  improve_prompt                          │
│  "원본 코드: ... 문제점: ... 개선하세요:" │
└─────────────────────────────────────────┘
                │
                ▼
┌─────────────────────────────────────────┐
│  llm → StrOutputParser()                 │
│  개선된 코드 생성                         │
└─────────────────────────────────────────┘
                │
                ▼
    "def calculate_average(numbers):
         if not numbers:
             return 0
         return sum(numbers) / len(numbers)"
```

---

##### 왜 체이닝이 유용한가?

1. **관심사 분리**: 문제점 분석과 코드 개선을 독립적인 단계로 분리
2. **재사용성**: `review_chain`을 다른 워크플로우에서도 재사용 가능
3. **디버깅 용이**: 각 단계의 입출력을 독립적으로 확인 가능
4. **유연한 구성**: 새로운 단계 추가(예: 테스트 코드 생성)가 쉬움

---

## 2. LiteLLM 소개

### 핵심 철학

> "효율성과 단순함을 위한 민첩하고 세련된 도구"

LiteLLM은 **경량화된 미니멀리스트 접근방식**의 LLM 통합 도구입니다. 100개 이상의 LLM을 **동일한 OpenAI 형식의 API**로 호출할 수 있게 해줍니다.

### 주요 특징

| 특징 | 설명 |
|------|------|
| **경량 아키텍처** | 복잡한 체인이나 모듈 없이 효율적이고 쉬운 통합 |
| **통합 인터페이스** | OpenAI, Azure, Anthropic, Cohere 등 여러 모델을 단일 API로 접근 |
| **간편한 통합** | 몇 줄의 코드로 기존 시스템에 빠르게 적용 가능 |
| **성능 최적화** | 최소한의 리소스 오버헤드로 모델 실행 |
| **편리한 인증** | 환경 변수 설정만으로 인증 및 연결 관리 |
| **빠른 프로토타이핑** | 간단한 API로 빠른 실험 및 개발 지원 |

### 지원하는 LLM 제공자

- OpenAI (GPT-4, GPT-4o, GPT-3.5)
- Anthropic (Claude 3.5, Claude 3)
- Google (Gemini)
- Azure OpenAI
- AWS Bedrock
- Cohere
- HuggingFace
- Ollama (로컬 모델)
- 그 외 100개 이상...

### 적합한 사용 사례

- 제한된 인프라의 웹 앱에서 AI 기능 구동
- 실시간, 저지연 LLM 애플리케이션
- 빠른 프로토타이핑 및 개발
- 여러 LLM 제공자 간 유연한 전환이 필요한 프로젝트
- 벤더 종속 방지가 필요한 경우

### LiteLLM 기본 사용 예제


```python
from litellm import completion

# OpenAI 모델 호출 (기본)
response = completion(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "리스트 컴프리헨션이란 무엇인가요? 간결하게 설명해주세요."}],
    temperature=0.7
)

print("⚡ LiteLLM 응답 (OpenAI):")
print(response.choices[0].message.content)
```



<div class="nb-output">

```text
⚡ LiteLLM 응답 (OpenAI):
리스트 컴프리헨션(List Comprehension)은 파이썬에서 리스트를 간결하고 효율적으로 생성하는 방법입니다. 기존 리스트나 반복 가능한 객체를 기반으로 새로운 리스트를 만들 수 있으며, 조건문을 통해 필터링도 가능합니다. 기본 구조는 다음과 같습니다:

```python
[expression for item in iterable if condition]
```

예를 들어, 0부터 9까지의 짝수 리스트를 생성하려면 다음과 같이 작성할 수 있습니다:

```python
even_numbers = [x for x in range(10) if x % 2 == 0]
```

위 코드는 `[0, 2, 4, 6, 8]`이라는 리스트를 생성합니다. 리스트 컴프리헨션을 사용하면 코드가 더 간결하고 가독성이 높아집니다.
```

</div>


### LiteLLM의 강점: 동일한 인터페이스로 다양한 모델 호출

LiteLLM의 가장 큰 장점은 모든 LLM을 **동일한 OpenAI 형식**으로 호출할 수 있다는 점입니다.


```python
from litellm import completion

# 다양한 모델을 동일한 인터페이스로 호출하는 예제
models_to_test = [
    "gpt-4o-mini",           # OpenAI
    # "claude-3-haiku-20240307",  # Anthropic (API 키 필요)
    # "gemini/gemini-pro",        # Google (API 키 필요)
]

question = "Python의 장점을 한 문장으로 설명해주세요."

for model in models_to_test:
    try:
        response = completion(
            model=model,
            messages=[{"role": "user", "content": question}],
            temperature=0.7
        )
        print(f"\n📌 {model}:")
        print(response.choices[0].message.content)
    except Exception as e:
        print(f"\n❌ {model}: {str(e)[:50]}...")
```


### LiteLLM 스트리밍 예제


```python
from litellm import completion

# 스트리밍 응답
print("⚡ LiteLLM 스트리밍 응답:\n")

response = completion(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "1부터 5까지 숫자를 세면서 각 숫자에 대한 재미있는 사실을 알려주세요."}],
    stream=True
)

for chunk in response:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="", flush=True)

print()  # 줄바꿈
```



<div class="nb-output">

```text
⚡ LiteLLM 스트리밍 응답:

물론입니다! 1부터 5까지 각 숫자에 대한 재미있는 사실을 소개할게요.

1. **1**: 숫자 1은 종종 '유일한 존재'를 나타내며, 수학적으로는 모든 수의 곱셈 항등원입니다. 즉, 어떤 숫자에 1을 곱해도 그 숫자 자신이 나옵니다. 또한, 여러 문화에서 중요한 상징적 의미를 가지며 '일체'나 '첫 번째'를 나타내기도 합니다.

2. **2**: 숫자 2는 대칭과 짝을 의미합니다. 대부분의 동물은 쌍으로 된 감각 기관(눈, 귀 등)을 가지고 있으며, 이는 생물학적으로 적응과 생존에 도움을 줍니다. 또한, '2'는 첫 번째 소수이기도 합니다.

3. **3**: 숫자 3은 매우 크고 다양한 의미를 지니고 있습니다. 서양 문화에서는 '세 가지 단위'가 여러 가지 것들(예: 과거, 현재, 미래; 정신, 육체, 영혼 등)을 상징합니다. 또한, 삼각형은 가장 안정적인 형체로 알려져 있습니다.

4. **4**: 숫자 4는 많은 문화에서 '완전함'을 상징합니다. 예를 들어, 동서양 모두에서 사계절(봄, 여름, 가을, 겨울)이라는 네 가지 주기가 중요시됩니다. 하지만, 일부 아시아 문화에서는 4가 '죽음'과 관련되어 있어 불행한 숫자로 여겨지기도 합니다.

5. **5**: 숫자 5는 인간의 손가락 수와 맞닿아 있어 많은 문화에서 특별한 중요성을 가집니다. 예를 들어, 다섯은 완전한 수로 간주되며, 자연에서 5각형(예: 별)이 자주 발견됩니다. 또한, 다섯은 '오감'을 대표하여 감각적인 경험과 관련이 깊습니다.

각 숫자는 단순한 수 이상의 의미와 상징을 지닙니다. 흥미로운 사실들로 여러분의 하루가 더욱 풍성해지기를 바랍니다!
```

</div>


---

## 3. LangChain vs LiteLLM 비교

### 비교 요약

| 항목 | LangChain | LiteLLM |
|------|-----------|---------|  
| **설계 철학** | 복잡성 & 커스터마이징 | 단순성 & 효율성 |
| **학습 곡선** | 높음 (모듈 구조 이해 필요) | 낮음 (초보자 친화적) |
| **사용 사례** | 복잡한 멀티스텝 워크플로우 | 빠른 프로토타이핑, 간단한 앱 |
| **아키텍처** | 모듈식, 체이닝 기반 | 스트림라인, 경량화 |
| **리소스 요구** | 상대적으로 높음 | 최소화된 오버헤드 |
| **주요 강점** | 유연성, 확장성, 다기능 | 속도, 간편함, 통합 API |

### 코드 복잡성 비교


```python
# 동일한 작업을 두 프레임워크로 구현

question = "Python에서 리스트와 튜플의 차이점은?"

# === LangChain 방식 ===
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

llm = ChatOpenAI(model="gpt-4o-mini")
prompt = ChatPromptTemplate.from_template("{question}")
chain = prompt | llm | StrOutputParser()
langchain_response = chain.invoke({"question": question})

print("🔗 LangChain (5줄의 설정 코드):")
print(langchain_response[:200] + "...\n")

# === LiteLLM 방식 ===
from litellm import completion

response = completion(model="gpt-4o-mini", messages=[{"role": "user", "content": question}])
litellm_response = response.choices[0].message.content

print("⚡ LiteLLM (1줄의 설정 코드):")
print(litellm_response[:200] + "...")
```



<div class="nb-output">

```text
🔗 LangChain (5줄의 설정 코드):
리스트와 튜플은 둘 다 파이썬에서 순서가 있는 컬렉션 데이터 타입이지만, 몇 가지 중요한 차이점이 있습니다. 아래에서 그 차이점을 설명합니다.

1. **변경 가능성 (Mutability)**:
   - **리스트**: 변경 가능합니다. 즉, 리스트의 요소를 추가, 삭제 또는 수정할 수 있습니다.
     ```python
     my_list = [1,...

⚡ LiteLLM (1줄의 설정 코드):
Python에서 리스트(`list`)와 튜플(`tuple`)은 둘 다 순서가 있는 컬렉션 타입이지만, 몇 가지 중요한 차이점이 있습니다.

1. **변경 가능성 (Mutability)**:
   - **리스트**: 변경 가능(mutable)합니다. 즉, 리스트의 요소를 추가, 삭제, 수정할 수 있습니다.
   - **튜플**: 변경 불가능(immutable...
```

</div>


### 언제 무엇을 선택할까?

#### LangChain을 선택하세요 if:

- ✅ 복잡한 멀티스텝 워크플로우가 필요한 경우
- ✅ 문서 처리, RAG, 에이전트 시스템을 구축하는 경우
- ✅ 다양한 컴포넌트의 세밀한 제어가 필요한 경우
- ✅ 외부 도구와의 깊은 통합이 필요한 경우

#### LiteLLM을 선택하세요 if:

- ✅ 간단하고 빠른 LLM 통합이 필요한 경우
- ✅ 복잡한 프레임워크 학습 없이 빠르게 시작하고 싶은 경우
- ✅ 리소스 제약이 있는 환경에서 작업하는 경우
- ✅ 여러 LLM 제공자 간 유연하게 전환하고 싶은 경우
- ✅ 신속한 배포가 우선인 경우

---

## 4. 두 프레임워크 함께 사용하기

두 도구는 **상호 배타적이지 않습니다**. 실제로 많은 프로젝트에서 함께 사용됩니다.

LangChain의 강력한 오케스트레이션 기능과 LiteLLM의 제공자 유연성을 결합하여 **두 세계의 장점**을 모두 활용할 수 있습니다.


```python
# LangChain 내에서 LiteLLM을 LLM Provider로 사용하는 예시
from langchain_community.chat_models import ChatLiteLLM
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# LiteLLM을 통해 모델 호출 (LangChain 체인 내에서)
chat = ChatLiteLLM(model="gpt-4o-mini", temperature=0.7)

prompt = ChatPromptTemplate.from_messages([
    ("system", "당신은 도움이 되는 AI 어시스턴트입니다."),
    ("user", "{question}")
])

chain = prompt | chat | StrOutputParser()

response = chain.invoke({"question": "LangChain과 LiteLLM을 함께 사용하는 장점은 무엇인가요?"})

print("🔗⚡ LangChain + LiteLLM 조합:")
print(response)
```


### 실전 예제: 모델 폴백(Fallback) 전략

LiteLLM의 모델 전환 기능을 활용하여, 주 모델이 실패할 경우 대체 모델을 사용하는 전략을 구현할 수 있습니다.


```python
from litellm import completion
import litellm

# 폴백 모델 목록
fallback_models = [
    "gpt-4o-mini",      # 1순위: GPT-4o-mini
    "gpt-3.5-turbo",    # 2순위: GPT-3.5
]

def completion_with_fallback(messages, **kwargs):
    """폴백 전략이 적용된 completion 함수"""
    for model in fallback_models:
        try:
            response = completion(
                model=model,
                messages=messages,
                **kwargs
            )
            print(f"✅ 성공: {model}")
            return response
        except Exception as e:
            print(f"❌ 실패 ({model}): {str(e)[:50]}")
            continue
    raise Exception("모든 모델이 실패했습니다.")

# 테스트
response = completion_with_fallback(
    messages=[{"role": "user", "content": "안녕하세요!"}],
    temperature=0.7
)
print(f"\n응답: {response.choices[0].message.content}")
```


---

## 5. 요약

### 핵심 정리

| 프레임워크 | 한 줄 요약 |
|-----------|----------|
| **LangChain** | 복잡한 LLM 워크플로우를 위한 풀스택 프레임워크 |
| **LiteLLM** | 100+ LLM을 단일 API로 호출하는 경량 라이브러리 |

### 선택 가이드

```
프로젝트 요구사항 분석
        │
        ▼
┌─────────────────────────────┐
│ 복잡한 체인/에이전트 필요?  │
└─────────────────────────────┘
        │
   Yes  │  No
        │
   ▼    │    ▼
LangChain    │
             │
     ┌───────────────────────┐
     │ 여러 LLM 제공자 전환?  │
     └───────────────────────┘
             │
        Yes  │  No
             │
        ▼    │    ▼
     LiteLLM │  직접 API 호출
```

### 결론

선택은 프로젝트의 **구체적인 요구사항**에 따라 달라집니다:

- **단순성과 빠른 개발**이 우선이라면 → **LiteLLM**
- **복잡한 워크플로우와 커스터마이징**이 필요하다면 → **LangChain**
- **두 가지 모두 필요하다면** → **함께 사용**

두 도구의 차이점을 이해하면 언어 모델 애플리케이션에 적합한 도구를 선택하는 데 도움이 됩니다.

---

## 참고 자료

- [LangChain 공식 문서](https://python.langchain.com/docs/)
- [LiteLLM 공식 문서](https://docs.litellm.ai/)
- [LangChain vs LiteLLM - Medium](https://medium.com/@heyamit10/langchain-vs-litellm-a9b784a2ad1a)
- [LangChain GitHub](https://github.com/langchain-ai/langchain)
- [LiteLLM GitHub](https://github.com/BerriAI/litellm)
