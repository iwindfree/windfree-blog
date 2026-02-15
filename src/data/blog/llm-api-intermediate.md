---
title: "LLM API 중급 (Part 2/3)"
author: iwindfree
pubDatetime: 2025-01-14T09:00:00Z
slug: "llm-api-intermediate"
category: "AI Engineering"
tags: ["ai", "llm", "api"]
description: "이 노트북은 LLM API 시리즈의 두 번째 파트로, 실무에서 필요한 다양한 기법들을 다룹니다."
---

이 노트북은 LLM API 시리즈의 두 번째 파트로, 실무에서 필요한 다양한 기법들을 다룹니다.

## 학습 목표

| 목표 | 설명 |
|------|------|
| 다양한 활용 | 코드 생성, 감정 분석 등 고급 활용 |
| API 파라미터 | temperature, max_tokens 등 제어 |
| 스트리밍 | 실시간 응답 출력 구현 |
| 에러 처리 | 안정적인 API 호출 패턴 |
| 다중 LLM | OpenAI, Claude, Ollama 비교 |
| 비용 관리 | 토큰 사용량 기반 비용 계산 |

## 시리즈 구성

- **Part 1**: LLM API 기초 - 환경설정, 메시지 구조, 기본 호출
- **Part 2 (현재)**: LLM API 중급 - 파라미터, 스트리밍, 에러처리, 다중 LLM
- **Part 3**: LLM API 고급 - 대화 이력, 캐싱, 에이전트, 프레임워크

## 사전 요구사항

- Part 1 완료
- OpenAI API 키 (필수)
- Anthropic API 키 (선택)
- Ollama 설치 (선택, 로컬 LLM용)


```python
import os
import time
from dotenv import load_dotenv
from openai import OpenAI, APIError, RateLimitError, AuthenticationError
from IPython.display import Markdown, display, update_display

load_dotenv(override=True)

# 클라이언트 초기화
client = OpenAI()
```


---

## 1. 다양한 활용 예시

Part 1에서 다룬 요약, 번역, Q&A 외에 더 다양한 활용 사례를 살펴봅니다.


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


### 1.1 코드 생성

프로그래밍 언어와 스타일을 지정하여 코드를 생성합니다.


```python
# 코드 생성
code_prompt = """당신은 Python 전문 프로그래머입니다.
요청된 기능을 구현하는 깔끔하고 효율적인 Python 코드를 작성해주세요.
코드에 간단한 docstring을 포함해주세요."""

code_request = "두 개의 리스트를 받아서 공통 요소만 반환하는 함수를 작성해주세요."

code = ask_llm(code_prompt, code_request)
print("=== 코드 생성 결과 ===")
display(Markdown(code))
```



<div class="nb-output">

```text
=== 코드 생성 결과 ===
<IPython.core.display.Markdown object>
```

</div>


### 1.2 감정 분석 (텍스트 분류)

텍스트를 정해진 카테고리로 분류합니다. 출력 형식을 명확히 지정하면 후처리가 쉬워집니다.


```python
# 감정 분석
sentiment_prompt = """당신은 감정 분석 전문가입니다.
주어진 텍스트의 감정을 분석하고 다음 형식으로만 응답하세요:
- 감정: [긍정/부정/중립]
- 신뢰도: [높음/중간/낮음]
- 근거: [1문장 설명]"""

reviews = [
    "이 제품 정말 최고예요! 배송도 빠르고 품질도 훌륭합니다.",
    "그냥 그래요. 나쁘진 않은데 특별히 좋지도 않네요.",
    "최악이에요. 돈 아깝습니다."
]

print("=== 감정 분석 결과 ===\n")
for review in reviews:
    result = ask_llm(sentiment_prompt, review)
    print(f'리뷰: "{review}"')
    print(result)
    print("-" * 50)
```



<div class="nb-output">

```text
=== 감정 분석 결과 ===

리뷰: "이 제품 정말 최고예요! 배송도 빠르고 품질도 훌륭합니다."
- 감정: 긍정
- 신뢰도: 높음
- 근거: 제품에 대한 높은 만족감과 긍정적인 경험이 명확하게 표현되고 있습니다.
--------------------------------------------------
리뷰: "그냥 그래요. 나쁘진 않은데 특별히 좋지도 않네요."
- 감정: 중립
- 신뢰도: 높음
- 근거: 긍정과 부정이 혼재되어 있지 않고, 평범함을 나타내는 표현이 사용되었다.
--------------------------------------------------
리뷰: "최악이에요. 돈 아깝습니다."
- 감정: 부정
- 신뢰도: 높음
- 근거: "최악이에요"와 "돈 아깝습니다"는 명백한 부정적인 감정을 표현하고 있습니다.
--------------------------------------------------
```

</div>


---

## 2. API 파라미터

Chat Completion API에서 자주 사용되는 파라미터들을 알아봅니다.

| 파라미터 | 설명 | 기본값 | 범위 |
|---------|------|--------|------|
| `model` | 사용할 모델 | 필수 | - |
| `messages` | 대화 메시지 배열 | 필수 | - |
| `temperature` | 창의성 조절 | 1.0 | 0.0~2.0 |
| `max_tokens` | 최대 출력 토큰 수 | 모델별 상이 | - |
| `top_p` | 누적 확률 기반 샘플링 | 1.0 | 0.0~1.0 |
| `stream` | 스트리밍 응답 | False | True/False |

### 2.1 temperature 파라미터

- **낮은 값 (0.0)**: 일관되고 결정론적인 응답
- **높은 값 (1.5+)**: 창의적이지만 예측하기 어려운 응답


```python
# temperature 비교
def compare_temperature(prompt: str):
    """temperature 값에 따른 응답 차이를 비교합니다."""
    for temp in [0.0, 1.0, 1.8]:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=temp,
            max_tokens=50
        )
        print(f"temperature={temp}: {response.choices[0].message.content}\n")

# 테스트
print("=== temperature 비교 ===")
compare_temperature("랜덤한 색상 하나를 추천해주세요.")
```



<div class="nb-output">

```text
=== temperature 비교 ===
temperature=0.0: 랜덤한 색상으로 "코발트 블루"를 추천합니다! 이 색상은 깊고 선명한 파란색으로, 시원하고 활기찬 느낌을 줍니다. 다양한 디자인에 잘 어울리

temperature=1.0: 랜덤한 색상으로 "청록색"을 추천합니다. 청록색은 청색과 녹색이 조화를 이루는 색상으로, 시원하고 자연적인 느낌을 줍니다. 디자인이나 인테리

temperature=1.8: 쇼킹 핑크(pure pink)를 추천드립니다! 신나는 에너지를 주는 색상으로, 다양한 요소와 잘 어울리며 튀는 느낌을 줄 수 있습니다. 또한, 시각적으로도 Legendary(백발
```

</div>


### 2.2 max_tokens 파라미터

출력의 최대 길이를 제한합니다. 비용 관리와 응답 시간 단축에 유용합니다.


```python
# max_tokens 비교
for max_tok in [20, 50, 100]:
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "파이썬의 장점을 설명해주세요."}],
        max_tokens=max_tok
    )
    content = response.choices[0].message.content
    print(f"max_tokens={max_tok}: {content[:80]}...")
    print(f"  (실제 출력: {response.usage.completion_tokens} 토큰)\n")
```



<div class="nb-output">

```text
max_tokens=20: 파이썬은 여러 가지 장점을 가진 널리 사용되는 프로그래밍 언어...
  (실제 출력: 20 토큰)

max_tokens=50: 파이썬은 다양한 이유로 인기가 높은 프로그래밍 언어입니다. 아래는 파이썬의 주요 장점들입니다:

1. **문법이 간단하고 가독성이 높음**: 파...
  (실제 출력: 50 토큰)

max_tokens=100: 파이썬은 다양한 장점 덕분에 많은 개발자와 기업에서 널리 사용되고 있는 프로그래밍 언어입니다. 다음은 파이썬의 주요 장점입니다:

1. **쉬운...
  (실제 출력: 100 토큰)
```

</div>


---

## 3. 스트리밍 응답

긴 응답의 경우 스트리밍을 사용하면 토큰이 생성되는 대로 실시간으로 출력받을 수 있습니다.

| 방식 | 특징 | 사용 사례 |
|------|------|----------|
| 일반 | 전체 응답 완료 후 수신 | 짧은 응답, 후처리 필요 시 |
| 스트리밍 | 토큰 단위로 실시간 수신 | 긴 응답, 챗봇 UI |


```python
def stream_response(user_message: str, system_prompt: str = "You are helpful."):
    """스트리밍 방식으로 응답을 받아 실시간 출력합니다."""
    stream = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message}
        ],
        stream=True
    )
    
    full_response = ""
    display_handle = display(Markdown(""), display_id=True)
    
    for chunk in stream:
        delta = chunk.choices[0].delta.content or ""
        full_response += delta
        update_display(Markdown(full_response), display_id=display_handle.display_id)
    
    return full_response
```



```python
# 스트리밍 테스트
print("=== 스트리밍 응답 ===")
_ = stream_response("파이썬의 장점을 3가지만 간단히 설명해주세요.")
```



<div class="nb-output">

```text
=== 스트리밍 응답 ===
<IPython.core.display.Markdown object>
```

</div>


---

## 4. 에러 처리

API 호출 시 다양한 오류가 발생할 수 있습니다.

| 에러 | 원인 | 해결책 |
|------|------|--------|
| `AuthenticationError` | 잘못된 API 키 | API 키 확인 |
| `RateLimitError` | 요청 한도 초과 | 대기 후 재시도 |
| `APIError` | 서버 오류 | 재시도 |
| `APIConnectionError` | 네트워크 오류 | 연결 확인 |


```python
def call_with_retry(messages: list, max_retries: int = 3):
    """에러 처리와 재시도 로직이 포함된 API 호출 함수"""
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages
            )
            return response.choices[0].message.content
        
        except AuthenticationError as e:
            print(f"인증 오류: API 키를 확인하세요.")
            raise  # 재시도 불필요
        
        except RateLimitError as e:
            wait_time = 2 ** attempt  # 지수 백오프: 1, 2, 4초
            print(f"요청 한도 초과. {wait_time}초 후 재시도... ({attempt + 1}/{max_retries})")
            time.sleep(wait_time)
        
        except APIError as e:
            print(f"API 오류: {e}")
            if attempt < max_retries - 1:
                time.sleep(1)
            else:
                raise
    
    raise Exception("최대 재시도 횟수 초과")

# 테스트
result = call_with_retry([{"role": "user", "content": "Hello!"}])
print(f"응답: {result}")
```


---

## 5. 다중 LLM 활용

여러 LLM 제공업체의 API를 비교해봅니다.

### 5.1 OpenAI vs Anthropic 비교

| 항목 | OpenAI | Anthropic |
|------|--------|----------|
| 라이브러리 | `openai` | `anthropic` |
| 메서드 | `chat.completions.create()` | `messages.create()` |
| system 전달 | messages 배열에 포함 | 별도 `system` 파라미터 |
| 응답 접근 | `response.choices[0].message.content` | `response.content[0].text` |


```python
import anthropic

# Anthropic 클라이언트 초기화
claude_client = anthropic.Anthropic()

def call_claude(system_prompt: str, user_message: str, model: str = "claude-sonnet-4-20250514"):
    """Anthropic Claude API를 호출합니다."""
    response = claude_client.messages.create(
        model=model,
        max_tokens=1024,
        system=system_prompt,  # system은 별도 파라미터
        messages=[{"role": "user", "content": user_message}]
    )
    return response

# Claude 호출 테스트
claude_response = call_claude(
    system_prompt="You are helpful. Respond in Korean.",
    user_message="파이썬의 장점을 2가지만 알려주세요."
)

print("=== Claude 응답 ===")
print(f"Content: {claude_response.content[0].text}")
print(f"\n--- 메타데이터 ---")
print(f"Model: {claude_response.model}")
print(f"Input: {claude_response.usage.input_tokens}, Output: {claude_response.usage.output_tokens} tokens")
```



<div class="nb-output">

```text
---------------------------------------------------------------------------
ConnectError                              Traceback (most recent call last)
File ~/workspace/ws.study/ai-engineering/.venv/lib/python3.13/site-packages/httpx/_transports/default.py:101, in map_httpcore_exceptions()
    100 try:
--> 101     yield
    102 except Exception as exc:

File ~/workspace/ws.study/ai-engineering/.venv/lib/python3.13/site-packages/httpx/_transports/default.py:250, in HTTPTransport.handle_request(self, request)
    249 with map_httpcore_exceptions():
--> 250     resp = self._pool.handle_request(req)
    252 assert isinstance(resp.stream, typing.Iterable)

File ~/workspace/ws.study/ai-engineering/.venv/lib/python3.13/site-packages/httpcore/_sync/connection_pool.py:256, in ConnectionPool.handle_request(self, request)
    255     self._close_connections(closing)
--> 256     raise exc from None
    258 # Return the response. Note that in this case we still have to manage
    259 # the point at which the response is closed.

File ~/workspace/ws.study/ai-engineering/.venv/lib/python3.13/site-packages/httpcore/_sync/connection_pool.py:236, in ConnectionPool.handle_request(self, request)
    234 try:
    235     # Send the request on the assigned connection.
--> 236     response = connection.handle_request(
    237         pool_request.request
    238     )
    239 except ConnectionNotAvailable:
    240     # In some cases a connection may initially be available to
    241     # handle a request, but then become unavailable.
    242     #
    243     # In this case we clear the connection and try again.

File ~/workspace/ws.study/ai-engineering/.venv/lib/python3.13/site-packages/httpcore/_sync/connection.py:101, in HTTPConnection.handle_request(self, request)
    100     self._connect_failed = True
--> 101     raise exc
    103 return self._connection.handle_request(request)

File ~/workspace/ws.study/ai-engineering/.venv/lib/python3.13/site-packages/httpcore/_sync/connection.py:78, in HTTPConnection.handle_request(self, request)
     77 if self._connection is None:
---> 78     stream = self._connect(request)
     80     ssl_object = stream.get_extra_info("ssl_object")

File ~/workspace/ws.study/ai-engineering/.venv/lib/python3.13/site-packages/httpcore/_sync/connection.py:156, in HTTPConnection._connect(self, request)
    155 with Trace("start_tls", logger, request, kwargs) as trace:
--> 156     stream = stream.start_tls(**kwargs)
    157     trace.return_value = stream

File ~/workspace/ws.study/ai-engineering/.venv/lib/python3.13/site-packages/httpcore/_backends/sync.py:154, in SyncStream.start_tls(self, ssl_context, server_hostname, timeout)
    150 exc_map: ExceptionMapping = {
    151     socket.timeout: ConnectTimeout,
    152     OSError: ConnectError,
    153 }
... (출력 172줄 생략)
```

</div>


### 5.2 Ollama 로컬 LLM

[Ollama](https://ollama.ai)를 사용하면 로컬에서 오픈소스 LLM을 실행할 수 있습니다.

| 장점 | 설명 |
|------|------|
| 무료 | API 비용 없음 |
| 프라이버시 | 데이터가 로컬에서만 처리 |
| 오프라인 | 인터넷 연결 불필요 |

```bash
# 설치 후 모델 다운로드
ollama pull llama3.2
ollama pull exaone3.5
```


```python
# Ollama 클라이언트 (OpenAI 호환 API)
ollama_client = OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="ollama"  # Ollama는 인증 불필요
)

def call_ollama(user_message: str, model: str = "exaone3.5"):
    """Ollama 로컬 LLM을 호출합니다."""
    try:
        response = ollama_client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": user_message}]
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"오류: {e} (Ollama가 실행 중인지 확인하세요)"

# Ollama 테스트 (Ollama 서버가 실행 중이어야 함)
ollama_result = call_ollama("What is Python? Answer in one sentence.")
print(f"=== Ollama 응답 ===")
print(ollama_result)
```



<div class="nb-output">

```text
=== Ollama 응답 ===
Python is a high-level programming language known for its readability and versatility, widely used for web development, data analysis, artificial intelligence, and more.
```

</div>


---

## 6. OpenAI 호환 인터페이스

OpenAI의 API 형식이 사실상 표준이 되어, 많은 제공업체가 호환 API를 제공합니다.
`base_url`만 변경하면 동일한 코드로 다양한 LLM에 접근할 수 있습니다.


```python
# 다양한 서비스 엔드포인트
ENDPOINTS = {
    "gemini": "https://generativelanguage.googleapis.com/v1beta/openai/",
    "groq": "https://api.groq.com/openai/v1",
    "ollama": "http://localhost:11434/v1",
}

# Gemini 클라이언트 예시
google_api_key = os.getenv("GOOGLE_API_KEY")
if google_api_key:
    gemini_client = OpenAI(
        api_key=google_api_key,
        base_url=ENDPOINTS["gemini"]
    )
    
    response = gemini_client.chat.completions.create(
        model="gemini-2.0-flash",
        messages=[{"role": "user", "content": "What is 2+2?"}]
    )
    print(f"Gemini 응답: {response.choices[0].message.content}")
else:
    print("GOOGLE_API_KEY가 설정되지 않았습니다.")
```



<div class="nb-output">

```text
GOOGLE_API_KEY가 설정되지 않았습니다.
```

</div>


---

## 7. 비용 계산

API 사용 시 토큰 수에 따라 비용이 발생합니다.

### 주요 모델 가격 (2025년 기준, 1M 토큰당)

| 모델 | Input | Output | 특징 |
|------|-------|--------|------|
| gpt-4o | $2.50 | $10.00 | 고성능 |
| gpt-4o-mini | $0.15 | $0.60 | 가성비 |
| claude-sonnet-4 | $3.00 | $15.00 | 균형 |

> 가격은 변동될 수 있으니 공식 문서를 확인하세요.


```python
# 비용 계산 함수
PRICING = {
    "gpt-4o": {"input": 2.50, "output": 10.00},
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "gpt-4.1": {"input": 2.00, "output": 8.00},
}

def calculate_cost(usage, model: str = "gpt-4o-mini") -> dict:
    """토큰 사용량을 기반으로 비용을 계산합니다."""
    if model not in PRICING:
        return {"error": f"Unknown model: {model}"}
    
    price = PRICING[model]
    input_cost = (usage.prompt_tokens / 1_000_000) * price["input"]
    output_cost = (usage.completion_tokens / 1_000_000) * price["output"]
    
    return {
        "input_tokens": usage.prompt_tokens,
        "output_tokens": usage.completion_tokens,
        "input_cost": input_cost,
        "output_cost": output_cost,
        "total_cost": input_cost + output_cost
    }
```



```python
# 비용 계산 테스트
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {"role": "system", "content": "You are helpful."},
        {"role": "user", "content": "파이썬의 장점을 5가지 알려주세요."}
    ]
)

cost = calculate_cost(response.usage, "gpt-4o-mini")
print("=== 비용 계산 결과 ===")
print(f"입력 토큰: {cost['input_tokens']:,}")
print(f"출력 토큰: {cost['output_tokens']:,}")
print(f"입력 비용: ${cost['input_cost']:.6f}")
print(f"출력 비용: ${cost['output_cost']:.6f}")
print(f"총 비용: ${cost['total_cost']:.6f}")
```


---

## 8. 요약

이번 노트북에서 학습한 내용:

| 주제 | 핵심 내용 |
|------|----------|
| **활용 예시** | 코드 생성, 감정 분석 등 system prompt로 역할 지정 |
| **파라미터** | temperature(창의성), max_tokens(길이) 등 제어 |
| **스트리밍** | stream=True로 실시간 응답 수신 |
| **에러 처리** | try-except와 지수 백오프로 안정성 확보 |
| **다중 LLM** | OpenAI, Claude, Ollama 상황에 맞게 선택 |
| **호환 API** | base_url 변경으로 다양한 LLM 접근 |
| **비용 관리** | 토큰 사용량 추적으로 비용 최적화 |

### 다음 단계

**Part 3 (고급)** 에서 다룰 내용:
- 대화 이력 관리 (멀티턴 대화)
- 프롬프트 캐싱
- LiteLLM 통합 인터페이스
- 다중 에이전트 시스템
- LangChain 프레임워크
