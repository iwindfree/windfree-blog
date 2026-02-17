---
title: "LLM API 기초 (Part 1/3)"
author: iwindfree
pubDatetime: 2025-01-13T09:00:00Z
slug: "llm-api-basic"
category: "LLM Engineering"
series: "LLM Engineering"
seriesOrder: 4
tags: ["ai", "llm", "api"]
description: "이 노트북은 LLM API 시리즈의 첫 번째 파트로, 대형 언어 모델 API의 기본 개념과 사용법을 다룹니다."
---

이 노트북은 LLM API 시리즈의 첫 번째 파트로, 대형 언어 모델 API의 기본 개념과 사용법을 다룹니다.

## 학습 목표

| 목표 | 설명 |
|------|------|
| LLM API 이해 | 대형 언어 모델 API의 개념과 구조 |
| 환경 설정 | API 키 발급 및 설정 방법 |
| 메시지 구조 | role과 messages 배열 이해 |
| 첫 API 호출 | OpenAI API 기본 사용법 |
| 기본 활용 | 요약, 번역, Q&A 예시 |

## 시리즈 구성

- **Part 1 (현재)**: LLM API 기초 - 환경설정, 메시지 구조, 기본 호출
- **Part 2**: LLM API 중급 - 파라미터, 스트리밍, 에러처리, 다중 LLM
- **Part 3**: LLM API 고급 - 대화 이력, 캐싱, 에이전트, 프레임워크

## 사전 요구사항

- Python 환경 설정 완료 (`0.environment.ipynb` 참조)
- OpenAI API 키 (필수)
- Anthropic API 키 (선택)

---

## 1. LLM API란?

**대형 언어 모델(LLM) API**는 HTTP 요청을 통해 AI 모델에 접근하는 인터페이스입니다. 
텍스트 생성, 요약, 번역, 코드 작성 등 다양한 작업을 수행할 수 있습니다.

### 주요 LLM 제공업체

| 제공업체 | 대표 모델 | 특징 |
|----------|----------|------|
| OpenAI | GPT-4o, GPT-4.1 | 업계 표준 API, 가장 널리 사용 |
| Anthropic | Claude 3.5/4 | 긴 컨텍스트(200K), 안전성 강조 |
| Google | Gemini 2.0 | 멀티모달, 무료 티어 제공 |
| Meta | Llama 3 | 오픈소스, 로컬 실행 가능 |

### API 호출 흐름

```
[사용자 코드] → HTTP POST 요청 → [LLM 서버] → JSON 응답 → [결과 처리]
```

---

## 2. 환경 설정

### API 키 발급

각 서비스에서 API 키를 발급받아야 합니다:

| 서비스 | 발급 링크 | 무료 티어 |
|--------|----------|----------|
| OpenAI | https://platform.openai.com | 유료 |
| Anthropic | https://console.anthropic.com | 유료 |
| Google AI | https://aistudio.google.com | 무료 티어 있음 |

### .env 파일 설정

프로젝트 루트에 `.env` 파일을 생성하고 API 키를 추가합니다:

```bash
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
GOOGLE_API_KEY=AI...
```

> **주의**: `.env` 파일은 절대 Git에 커밋하지 마세요! `.gitignore`에 추가하세요.


```python
# 필요한 라이브러리 설치 (필요시 주석 해제)
# !pip install openai anthropic python-dotenv
```



```python
import os
from dotenv import load_dotenv
from openai import OpenAI
from IPython.display import Markdown, display

# 환경변수 로드
load_dotenv(override=True)
```



<div class="nb-output">

```text
True
```

</div>



```python
# API 키 확인 함수
def check_api_key(name: str, key: str | None, prefix_len: int = 5) -> None:
    """API 키 존재 여부를 확인하고 앞부분만 출력합니다."""
    if key:
        print(f"  {name}: {key[:prefix_len]}...")
    else:
        print(f"  {name}: 설정되지 않음")

# API 키 로드 및 확인
openai_api_key = os.getenv("OPENAI_API_KEY")
anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
google_api_key = os.getenv("GOOGLE_API_KEY")

print("=== API 키 상태 ===")
check_api_key("OPENAI_API_KEY", openai_api_key)
check_api_key("ANTHROPIC_API_KEY", anthropic_api_key)
check_api_key("GOOGLE_API_KEY", google_api_key)
```



<div class="nb-output">

```text
=== API 키 상태 ===
  OPENAI_API_KEY: sk-pr...
  ANTHROPIC_API_KEY: sk-an...
  GOOGLE_API_KEY: AIzaS...
```

</div>



```python
# OpenAI 클라이언트 초기화
client = OpenAI()  # OPENAI_API_KEY 환경변수를 자동으로 사용
```


---

## 3. 메시지 구조 이해

LLM API 를 사용하여 전송되는 메세지는  일반적으로 아래와 같이  **메시지 배열** 형태로 구성됩니다..

### 기본 구조

```json
{
  "model": "gpt-4o-mini",
  "messages": [
    {"role": "system", "content": "당신은 친절한 AI입니다."},
    {"role": "user", "content": "안녕하세요!"},
    {"role": "assistant", "content": "안녕하세요! 무엇을 도와드릴까요?"},
    {"role": "user", "content": "파이썬에 대해 알려주세요."}
  ]
}
```

### 메시지 역할(Role)

| 역할 | 설명 | 필수 여부 |
|------|------|----------|
| **system** | AI의 행동 지침, 페르소나 설정 | 선택 |
| **user** | 사용자의 질문이나 요청 | 필수 |
| **assistant** | AI의 이전 응답 (대화 이력) | 선택 |

### 핵심 포인트

- `system` 메시지로 AI의 역할과 행동 방식을 지정
- 대화 맥락을 유지하려면 이전 메시지들을 배열에 포함
- LLM은 상태를 저장하지 않으므로, 매 요청마다 전체 대화 전달 필요


```python
# 메시지 구조 예시
messages = [
    {"role": "system", "content": "당신은 친절한 AI 어시스턴트입니다. 간결하게 답변하세요."},
    {"role": "user", "content": "파이썬이란 무엇인가요?"}
]

# 메시지 구조 확인
print("=== 메시지 구조 ===")
for i, msg in enumerate(messages):
    print(f"[{i}] role: {msg['role']}, content: {msg['content'][:30]}...")
```



<div class="nb-output">

```text
=== 메시지 구조 ===
[0] role: system, content: 당신은 친절한 AI 어시스턴트입니다. 간결하게 답변하세...
[1] role: user, content: 파이썬이란 무엇인가요?...
```

</div>


---

## 4. 첫 번째 API 호출

이제 실제로 OpenAI API를 호출해봅니다.


```python
# 기본 API 호출
client = OpenAI()
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {"role": "system", "content": "당신은 친절한 AI 어시스턴트입니다."},
        {"role": "user", "content": "안녕하세요! 간단히 자기소개 해주세요."}
    ]
)

# 응답 출력
print("=== AI 응답 ===")
print(response.choices[0].message.content)
```



<div class="nb-output">

```text
=== AI 응답 ===
안녕하세요! 저는 AI 어시스턴트입니다. 다양한 질문에 답하고, 정보를 제공하며, 여러 주제에 대해 대화할 수 있도록 도와드립니다. 필요하신 내용이 있으면 언제든지 말씀해 주세요!
```

</div>


### 응답 객체 구조

API 응답에는 생성된 텍스트 외에도 다양한 메타데이터가 포함됩니다.

| 필드 | 설명 |
|------|------|
| `id` | 요청 고유 식별자 |
| `model` | 실제 사용된 모델명 |
| `choices[0].message.content` | 생성된 응답 텍스트 |
| `choices[0].finish_reason` | 종료 이유 (stop, length 등) |
| `usage.prompt_tokens` | 입력 토큰 수 |
| `usage.completion_tokens` | 출력 토큰 수 |


```python
# 응답 객체 상세 확인
print("=== 응답 객체 상세 ===")
print(f"ID: {response.id}")
print(f"Model: {response.model}")
print(f"Finish Reason: {response.choices[0].finish_reason}")
print(f"\n--- 토큰 사용량 ---")
print(f"입력 토큰: {response.usage.prompt_tokens}")
print(f"출력 토큰: {response.usage.completion_tokens}")
print(f"총 토큰: {response.usage.total_tokens}")
```



<div class="nb-output">

```text
=== 응답 객체 상세 ===
ID: chatcmpl-D7wAcrmOpfFrh7QH7AD9qSP4OXYxf
Model: gpt-4o-mini-2024-07-18
Finish Reason: stop

--- 토큰 사용량 ---
입력 토큰: 36
출력 토큰: 50
총 토큰: 86
```

</div>



```python
# 전체 응답을 JSON으로 확인
print("=== 전체 응답 (JSON) ===")
print(response.model_dump_json(indent=2))
```



<div class="nb-output">

```text
=== 전체 응답 (JSON) ===
{
  "id": "chatcmpl-D7wAcrmOpfFrh7QH7AD9qSP4OXYxf",
  "choices": [
    {
      "finish_reason": "stop",
      "index": 0,
      "logprobs": null,
      "message": {
        "content": "안녕하세요! 저는 AI 어시스턴트입니다. 다양한 질문에 답하고, 정보를 제공하며, 여러 주제에 대해 대화할 수 있도록 도와드립니다. 필요하신 내용이 있으면 언제든지 말씀해 주세요!",
        "refusal": null,
        "role": "assistant",
        "annotations": [],
        "audio": null,
        "function_call": null,
        "tool_calls": null
      }
    }
  ],
  "created": 1770784274,
  "model": "gpt-4o-mini-2024-07-18",
  "object": "chat.completion",
  "service_tier": "default",
  "system_fingerprint": "fp_f4ae844694",
  "usage": {
    "completion_tokens": 50,
    "prompt_tokens": 36,
    "total_tokens": 86,
    "completion_tokens_details": {
      "accepted_prediction_tokens": 0,
      "audio_tokens": 0,
      "reasoning_tokens": 0,
      "rejected_prediction_tokens": 0
    },
    "prompt_tokens_details": {
      "audio_tokens": 0,
      "cached_tokens": 0
    }
  }
}
```

</div>


---

## 5. 기본 활용 예시

LLM API의 대표적인 활용 사례를 살펴봅니다.

### 활용 사례 요약

| 활용 사례 | system prompt 역할 | 특징 |
|----------|-------------------|------|
| 텍스트 요약 | 요약 전문가 역할 | 핵심 내용 추출 |
| 번역 | 번역가 역할 | 문맥 유지 |
| Q&A | 특정 분야 전문가 | 정확한 답변 |


```python
# 공통 헬퍼 함수
def ask_llm(system_prompt: str, user_message: str, model: str = "gpt-4o-mini") -> str:
    """시스템 프롬프트와 사용자 메시지로 LLM을 호출합니다."""
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message}
        ]
    )
    return response.choices[0].message.content
```


### 5.1 텍스트 요약

긴 텍스트에서 핵심 내용만 추출합니다.


```python
# 텍스트 요약
summarize_prompt = """당신은 텍스트 요약 전문가입니다.
주어진 텍스트를 3문장 이내로 핵심만 요약해주세요."""

long_text = """
인공지능(AI)은 인간의 학습능력, 추론능력, 지각능력, 자연언어 이해능력 등을 
컴퓨터 프로그램으로 실현한 기술입니다. 최근 딥러닝의 발전으로 AI는 이미지 인식, 
자연어 처리, 게임, 자율주행 등 다양한 분야에서 인간을 능가하는 성능을 보여주고 있습니다.
특히 GPT, Claude 등 대형 언어 모델(LLM)의 등장으로 AI는 글쓰기, 코딩, 분석 등 
지식 노동 영역에서도 활용되고 있습니다.
"""

summary = ask_llm(summarize_prompt, long_text)
print("=== 요약 결과 ===")
print(summary)
```



<div class="nb-output">

```text
=== 요약 결과 ===
인공지능(AI)은 학습, 추론, 지각 및 언어 이해 능력을 컴퓨터 프로그램으로 구현한 기술이다. 최근 딥러닝 발전 덕분에 AI는 이미지 인식, 자연어 처리 등에서 뛰어난 성능을 보이고 있다. 특히 대형 언어 모델(LLM)의 등장으로 AI는 글쓰기, 코딩, 분석 등 지식 노동 분야에서도 활용되고 있다.
```

</div>


### 5.2 번역

한국어를 영어로 자연스럽게 번역합니다.


```python
# 번역
translate_prompt = """당신은 전문 번역가입니다.
한국어를 영어로 자연스럽게 번역해주세요.
직역보다는 의역을 선호하며, 원문의 뉘앙스를 살려주세요."""

korean_text = "오늘 날씨가 정말 좋네요. 산책하기 딱 좋은 날이에요!"

translation = ask_llm(translate_prompt, korean_text)
print("=== 번역 결과 ===")
print(f"원문: {korean_text}")
print(f"번역: {translation}")
```



<div class="nb-output">

```text
=== 번역 결과 ===
원문: 오늘 날씨가 정말 좋네요. 산책하기 딱 좋은 날이에요!
번역: The weather is absolutely wonderful today. It's a perfect day for a walk!
```

</div>


### 5.3 Q&A (질문-답변)

특정 분야의 전문가로 설정하여 질문에 답변합니다.


```python
# Q&A
qa_prompt = """당신은 Python 프로그래밍 튜터입니다.
초보자도 이해할 수 있도록 쉽고 명확하게 설명해주세요.
필요하면 간단한 예시 코드도 포함해주세요."""

question = "파이썬에서 리스트와 튜플의 차이점이 뭔가요?"

answer = ask_llm(qa_prompt, question)
print("=== Q&A 결과 ===")
print(f"Q: {question}\n")
display(Markdown(answer))
```



<div class="nb-output">

```text
=== Q&A 결과 ===
Q: 파이썬에서 리스트와 튜플의 차이점이 뭔가요?
<IPython.core.display.Markdown object>
```

</div>


---

## 6. 실습: 나만의 프롬프트 만들기

아래 셀을 수정하여 다양한 system prompt를 실험해보세요.


```python
# 실습: 나만의 프롬프트
my_system_prompt = """당신은 _____입니다.
_____ 방식으로 답변해주세요."""

my_question = "여기에 질문을 입력하세요"

# 실행
# result = ask_llm(my_system_prompt, my_question)
# display(Markdown(result))
```


---

## 7. 요약

이번 노트북에서 학습한 내용:

| 주제 | 핵심 내용 |
|------|----------|
| **LLM API** | HTTP 요청으로 AI 모델에 접근하는 인터페이스 |
| **환경 설정** | .env 파일에 API 키 저장, python-dotenv로 로드 |
| **메시지 구조** | system/user/assistant 역할로 대화 구성 |
| **응답 객체** | choices[0].message.content로 텍스트, usage로 토큰 확인 |
| **활용 예시** | 요약, 번역, Q&A 등 system prompt로 역할 지정 |

### 다음 단계

**Part 2 (중급)** 에서 다룰 내용:
- API 파라미터 (temperature, max_tokens 등)
- 스트리밍 응답
- 에러 처리
- 다중 LLM 비교 (OpenAI vs Claude vs Ollama)
- 비용 계산
