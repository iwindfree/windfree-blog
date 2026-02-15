---
title: "LLM API 고급 (Part 3/3)"
author: iwindfree
pubDatetime: 2025-01-15T09:00:00Z
slug: "llm-api-advanced"
category: "AI Engineering"
tags: ["ai", "llm", "api"]
description: "이 노트북은 LLM API 시리즈의 마지막 파트로, 프로덕션 수준의 고급 기법들을 다룹니다."
---

이 노트북은 LLM API 시리즈의 마지막 파트로, 프로덕션 수준의 고급 기법들을 다룹니다.

## 학습 목표

| 목표 | 설명 |
|------|------|
| 대화 이력 관리 | 멀티턴 대화 시스템 구현 |
| 추론 능력 테스트 | 논리 퍼즐로 모델 성능 비교 |
| 프롬프트 캐싱 | 비용 절감 기법 |
| LiteLLM | 100+ LLM 통합 인터페이스 |
| 다중 에이전트 | 여러 AI가 협업하는 시스템 |
| LangChain | LLM 애플리케이션 프레임워크 |

## 시리즈 구성

- **Part 1**: LLM API 기초 - 환경설정, 메시지 구조, 기본 호출
- **Part 2**: LLM API 중급 - 파라미터, 스트리밍, 에러처리, 다중 LLM
- **Part 3 (현재)**: LLM API 고급 - 대화 이력, 캐싱, 에이전트, 프레임워크

## 사전 요구사항

- Part 1, 2 완료
- OpenAI API 키 (필수)
- Anthropic, Google API 키 (선택)
- `litellm`, `langchain-openai` 설치


```python
# 필요한 라이브러리 설치 (필요시 주석 해제)
#pip install litellm langchain-openai
```



<div class="nb-output">

```text
  Cell In[32], line 2
    pip install litellm langchain-openai
        ^
SyntaxError: invalid syntax
```

</div>



```python
import os
from dotenv import load_dotenv
from openai import OpenAI
from IPython.display import Markdown, display

load_dotenv(override=True)

# 클라이언트 초기화
client = OpenAI()
```


---

## 1. 대화 이력 관리

LLM API는 상태를 유지하지 않습니다. 대화의 맥락을 유지하려면 이전 메시지들을 함께 전송해야 합니다.

### 핵심 개념

```
요청 1: [system, user1] → assistant1
요청 2: [system, user1, assistant1, user2] → assistant2
요청 3: [system, user1, assistant1, user2, assistant2, user3] → assistant3
```

간단한 예를 들어 설명하겠습니다. 아래의 코드를 수행해보면 재미있는 현상을 발견할 수 있습니다.


```python
from openai import OpenAI
openai_client = OpenAI()
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Hello, My name is windfree."}]
response = openai_client.chat.completions.create(
    model="gpt-4",
    messages=messages,)
print(response.choices[0].message.content)
```



<div class="nb-output">

```text
Hello, Windfree! How can I assist you today?
```

</div>



```python
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What is my name?"}]
response = openai_client.chat.completions.create(
    model="gpt-4",
    messages=messages,)
print(response.choices[0].message.content)
```



<div class="nb-output">

```text
I'm sorry, but as an AI, I don't have access to personal data about individuals unless it has been shared with me in the course of our conversation. I'm designed to respect user privacy and confidentiality.
```

</div>


첫번째 호출에서 내 이름을 말해준 후에 두번째 호출에서 내 이름을 물어보았을 때 LLM 은 내 이름을 모른다는 답을 하고 있습니다. 이유가 뭘까요? LLM 에 대한 모든 호출은 완전히 Stateless 한 상태입니다. 매번 완전히 새로운 호출인 셈이죠. LLM 이 “기억” 을 가진 것처럼 만드는 것은 AI 개발자의 몫입니다.


```python
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Hello, My name is windfree."},
    {"role": "assistant", "content": "Hello, Windfree! How can I assist you today?"},
    {"role": "user", "content": "What is my name?"}]
response = openai_client.chat.completions.create(
    model="gpt-4",messages=messages)
print(response.choices[0].message.content)
```



<div class="nb-output">

```text
Your name is Windfree.
```

</div>


당연한 얘기일 수 있지만,  정리해보면:

 * LLM에 대한 모든 호출은 무상태(stateless)다.
 * 매번 지금까지의 전체 대화를 입력 프롬프트에 담아 전달한다.
 * 이게 LLM이 기억을 가진 것 같은 착각을 만든다 — 대화 맥락을 유지하는 것처럼 보이게 하지만 이건 트릭이다.
 * 매번 전체 대화를 제공한 결과일 뿐 LLM은 그저 시퀀스에서 다음에 올 가장 가능성 높은 토큰을 예측할 뿐이다.
 * 시퀀스에 “내 이름은 windfree야”가 있고 나중에 “내 이름이 뭐지?”라고 물으면… windfree라고 예측하는 것!

많은 제품들이 정확히 이 트릭을 사용합니다. 메시지를 보낼 때마다 전체 대화가 함께 전달되는 겁니다. “그러면 매번 이전 대화 전체에 대해 추가 비용을 내야 하는 건가요?” 네. 당연히 그렇습니다. 그리고 그게 우리가 원하는 것이기도 합니다. 우리는 LLM이 전체 대화를 되돌아보며 다음 토큰을 예측하길 기대하고 있는 상태이며 그에 대한 사용료를 내야 하는 것입니다.

실제로 LLM API를 다뤄보셨으니 체감하시겠지만, 매 요청마다 이전 대화 내역을 messages 배열에 다시 담아 보내는 구조가 바로 이 무상태성 때문입니다. 흔히 사용하는 “기억” 구현 기법들은 아래와 같습니다.

 * 컨텍스트 주입: 이전 대화를 messages에 누적
 * 요약/압축: 긴 대화는 요약해서 system prompt에 삽입
 * RAG: 외부 저장소에서 관련 정보 검색 후 주입
 * 메모리 DB: 사용자별 중요 정보를 별도 저장 후 필요시 주입
 
API 요금 구조를 보면 input token과 output token을 따로 과금하는데, 대화가 길어질수록 input token이 누적되어 비용이 기하급수적으로 늘어납니다. 그래서 실무에서는 대화 요약, sliding window, 오래된 메시지 삭제 같은 전략을 쓰게 됩니다.

이제 좀 더 실용적인 예제를 살펴보도록 하겠습니다.


```python
from IPython.display import Markdown, display, update_display
from typing import Generator

# 대화 이력 관리 클래스
class ChatSession:
    """대화 이력을 관리하는 채팅 세션 클래스"""

    def __init__(self, system_prompt: str = "", model: str = "gpt-4o-mini"):
        self.model = model
        self.messages = []
        self.total_tokens = 0

        if system_prompt:
            self.messages.append({"role": "system", "content": system_prompt})

    def chat(self, user_input: str, stream: bool = False):
        """사용자 입력을 받아 응답을 반환합니다.

        Args:
            user_input: 사용자 입력 메시지
            stream: True면 스트리밍 모드로 실시간 출력

        Returns:
            stream=False: 전체 응답 문자열
            stream=True: 실시간 출력 후 전체 응답 문자열 반환
        """
        self.messages.append({"role": "user", "content": user_input})

        if stream:
            return self._chat_stream()
        else:
            return self._chat_normal()

    def _chat_normal(self) -> str:
        """일반 모드로 응답을 받습니다."""
        response = client.chat.completions.create(
            model=self.model,
            messages=self.messages,
        )

        assistant_reply = response.choices[0].message.content
        self.messages.append({"role": "assistant", "content": assistant_reply})
        self.total_tokens += response.usage.total_tokens

        return assistant_reply

    def _chat_stream(self) -> str:
        """스트리밍 모드로 응답을 받아 실시간 출력합니다."""
        response = client.chat.completions.create(
            model=self.model,
            messages=self.messages,
            stream=True
        )

        full_response = ""
        display_handle = display(Markdown(""), display_id=True)

        for chunk in response:
            delta = chunk.choices[0].delta.content or ""
            full_response += delta
            update_display(Markdown(full_response), display_id=display_handle.display_id)

        # 대화 이력에 추가
        self.messages.append({"role": "assistant", "content": full_response})

        return full_response

    def chat_generator(self, user_input: str) -> Generator[str, None, None]:
        """스트리밍 응답을 제너레이터로 반환합니다 (Gradio 등에서 활용).

        Args:
            user_input: 사용자 입력 메시지

        Yields:
            토큰 단위로 누적된 응답 문자열
        """
        self.messages.append({"role": "user", "content": user_input})

        response = client.chat.completions.create(
            model=self.model,
            messages=self.messages,
            stream=True
        )

        full_response = ""
        for chunk in response:
            delta = chunk.choices[0].delta.content or ""
            full_response += delta
            yield full_response

        # 대화 이력에 추가
        self.messages.append({"role": "assistant", "content": full_response})

    def show_history(self):
        """대화 이력을 출력합니다."""
        icons = {"system": "⚙️", "user": "👤", "assistant": "🤖"}
        for msg in self.messages:
            icon = icons.get(msg["role"], "❓")
            content = msg["content"][:80] + "..." if len(msg["content"]) > 80 else msg["content"]
            print(f"{icon} [{msg['role']}]: {content}")

    def get_stats(self) -> dict:
        """세션 통계를 반환합니다."""
        return {
            "message_count": len(self.messages),
            "total_tokens": self.total_tokens
        }
```



```python
# 세션 테스트
session = ChatSession(
    system_prompt="당신은 파이썬 튜터입니다. 초보자에게 친절하게 설명해주세요.",
    model="gpt-4o-mini"
)

# 첫 번째 질문
print("=== 첫 번째 질문 ===")
reply1 = session.chat("파이썬에서 리스트 컴프리헨션이 뭔가요?")
display(Markdown(reply1))
```



<div class="nb-output">

```text
=== 첫 번째 질문 ===
<IPython.core.display.Markdown object>
```

</div>



```python
# 후속 질문 (맥락 유지)
print("=== 후속 질문 (맥락 유지) ===")
reply2 = session.chat("그거랑 map 함수랑 뭐가 다른가요?")
display(Markdown(reply2))
```



<div class="nb-output">

```text
=== 후속 질문 (맥락 유지) ===
<IPython.core.display.Markdown object>
```

</div>



```python
# 대화 이력 및 통계
print("\n=== 대화 이력 ===")
session.show_history()

print(f"\n=== 통계 ===")
stats = session.get_stats()
print(f"메시지 수: {stats['message_count']}")
print(f"총 토큰: {stats['total_tokens']}")
```



```python
# chat_generator를 IPython에서 사용하는 예제
from IPython.display import Markdown, display, update_display

# 새 세션 생성
stream_session = ChatSession(
    system_prompt="당신은 친절한 AI입니다. 간결하게 답변해주세요.",
    model="gpt-4o-mini"
)

# chat_generator로 스트리밍 출력
print("=== chat_generator 사용 예제 ===")
display_handle = display(Markdown(""), display_id=True)

for partial_response in stream_session.chat_generator("파이썬의 장점 3가지를 알려주세요"):
    # partial_response는 지금까지 누적된 응답
    update_display(Markdown(partial_response), display_id=display_handle.display_id)
```



<div class="nb-output">

```text
=== chat_generator 사용 예제 ===
<IPython.core.display.Markdown object>
```

</div>


### 스트리밍과 Generator 패턴

`ChatSession` 클래스는 스트리밍 응답을 위한 두 가지 방식을 제공합니다:

| 메서드 | 반환 타입 | 사용 환경 |
|--------|----------|----------|
| `chat(msg, stream=True)` | `str` | Jupyter Notebook (자동 출력) |
| `chat_generator(msg)` | `Generator` | Gradio, FastAPI 등 (직접 제어) |

### yield와 Generator란?

Python의 `yield` 키워드는 함수를 **제너레이터(Generator)** 로 만듭니다. 일반 함수는 `return`으로 값을 한 번에 반환하지만, 제너레이터는 `yield`로 값을 **하나씩 순차적으로** 반환합니다.

```python
# 일반 함수: 모든 값을 한 번에 반환
def get_all():
    return [1, 2, 3]  # 메모리에 전체 리스트 생성

# 제너레이터: 값을 하나씩 반환
def get_one_by_one():
    yield 1  # 첫 번째 호출에서 반환
    yield 2  # 두 번째 호출에서 반환
    yield 3  # 세 번째 호출에서 반환
```

**스트리밍에서의 장점:**
- 전체 응답을 기다리지 않고 토큰이 생성되는 즉시 처리 가능
- 메모리 효율적 (전체 응답을 한번에 저장하지 않음)
- Gradio, FastAPI 등 프레임워크와 자연스럽게 통합

---

## 2. 추론 능력 테스트

논리 퍼즐로 다양한 모델의 추론 능력을 비교해봅니다.


```python
# 확률 문제
probability_puzzle = [
    {"role": "user", "content": 
     """동전 2개를 던졌습니다. 그 중 하나가 앞면이라는 것을 알게 되었습니다.
     나머지 하나가 뒷면일 확률은 얼마일까요?
     
     힌트: 이것은 조건부 확률 문제입니다. 단순히 1/2가 아닙니다.
     단계별로 풀이해주세요."""}
]

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=probability_puzzle
)

print("=== 확률 퍼즐 (GPT-4o-mini) ===")
display(Markdown(response.choices[0].message.content))
```



<div class="nb-output">

```text
=== 확률 퍼즐 (GPT-4o-mini) ===
<IPython.core.display.Markdown object>
```

</div>



```python
bookworm_puzzle = [
           {"role": "user", "content":
            """책장에 2권짜리 시리즈가 나란히 놓여 있습니다.
            각 책의 본문 두께는 3cm이고, 앞뒤 표지는 각각 3mm입니다.

            책벌레가 1권의 첫 페이지부터 2권의 마지막 페이지까지
            수직으로 뚫고 지나갔습니다.

            책벌레가 이동한 거리는 몇 cm일까요?

            (힌트: 책이 책장에 어떻게 놓이는지 시각화해보세요)"""}
       ]

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=bookworm_puzzle
)

print("=== 책벌레 퍼즐 (GPT-4o-mini) ===")
display(Markdown(response.choices[0].message.content))
```



<div class="nb-output">

```text
=== 책벌레 퍼즐 (GPT-4o-mini) ===
<IPython.core.display.Markdown object>
```

</div>


---

## 3. LiteLLM 통합 인터페이스

LiteLLM은 100개 이상의 LLM을 단일 인터페이스로 호출할 수 있게 해주는 라이브러리입니다.

### 장점

- 통일된 API로 다양한 모델 접근
- 비용 추적 기능 내장
- Fallback/Retry 로직 지원


```python
from litellm import completion
# 다양한 모델 호출
test_message = [{"role": "user", "content": "What is 2+2? Answer with just the number."}]

# OpenAI
response = completion(model="openai/gpt-4o-mini", messages=test_message)
print(f"GPT-4o-mini: {response.choices[0].message.content}")
print(f"  토큰: {response.usage.total_tokens}, 비용: ${response._hidden_params.get('response_cost', 0):.6f}")
```



<div class="nb-output">

```text
GPT-4o-mini: 4
  토큰: 21, 비용: $0.000004
```

</div>



```python
#pip install pip-system-certs
```



<div class="nb-output">

```text
Collecting pip-system-certs
  Downloading pip_system_certs-5.3-py3-none-any.whl.metadata (3.9 kB)
Requirement already satisfied: pip>=24.2 in /Users/windfree/workspace/ws.study/ai-engineering/.venv/lib/python3.13/site-packages (from pip-system-certs) (24.3.1)
Downloading pip_system_certs-5.3-py3-none-any.whl (6.9 kB)
Installing collected packages: pip-system-certs
Successfully installed pip-system-certs-5.3

[1m[[0m[34;49mnotice[0m[1;39;49m][0m[39;49m A new release of pip is available: [0m[31;49m24.3.1[0m[39;49m -> [0m[32;49m26.0.1[0m
[1m[[0m[34;49mnotice[0m[1;39;49m][0m[39;49m To update, run: [0m[32;49mpip3 install --upgrade pip[0m
Note: you may need to restart the kernel to use updated packages.
```

</div>


아래 예제에서 SSL 오류가 나는 경우  
* pip install pip-system-certs 을 실행해 줍니다. 
* pip 모듈이 없다고 나오는 경우에는 터미널에서 .venv/bin/python -m ensurepip --upgrade 를 실행하고 커널을 재실행해주세요.


```python
# Anthropic (LiteLLM 통해)
response = completion(model="anthropic/claude-sonnet-4-20250514", messages=test_message)
print(f"Claude Sonnet: {response.choices[0].message.content}")
print(f"  토큰: {response.usage.total_tokens}, 비용: ${response._hidden_params.get('response_cost', 0):.6f}")
```



<div class="nb-output">

```text
Claude Sonnet: 4
  토큰: 25, 비용: $0.000135
```

</div>



```python
# Gemini (LiteLLM 통해)
response = completion(model="gemini/gemini-2.0-flash", messages=test_message)
print(f"Gemini 2.0 Flash: {response.choices[0].message.content}")
print(f"  토큰: {response.usage.total_tokens}, 비용: ${response._hidden_params.get('response_cost', 0):.6f}")
```



<div class="nb-output">

```text

[1;31mGive Feedback / Get Help: https://github.com/BerriAI/litellm/issues/new[0m
LiteLLM.Info: If you need to debug this error, use `litellm._turn_on_debug()'.
---------------------------------------------------------------------------
HTTPStatusError                           Traceback (most recent call last)
File ~/workspace/ws.study/ai-engineering/.venv/lib/python3.13/site-packages/litellm/llms/vertex_ai/gemini/vertex_and_google_ai_studio_gemini.py:2599, in VertexLLM.completion(self, model, messages, model_response, print_verbose, custom_llm_provider, encoding, logging_obj, optional_params, acompletion, timeout, vertex_project, vertex_location, vertex_credentials, gemini_api_key, litellm_params, logger_fn, extra_headers, client, api_base)
   2598 try:
-> 2599     response = client.post(url=url, headers=headers, json=data, logging_obj=logging_obj)  # type: ignore
   2600     response.raise_for_status()

File ~/workspace/ws.study/ai-engineering/.venv/lib/python3.13/site-packages/litellm/llms/custom_httpx/http_handler.py:979, in HTTPHandler.post(self, url, data, json, params, headers, stream, timeout, files, content, logging_obj)
    978     setattr(e, "status_code", e.response.status_code)
--> 979     raise e
    980 except Exception as e:

File ~/workspace/ws.study/ai-engineering/.venv/lib/python3.13/site-packages/litellm/llms/custom_httpx/http_handler.py:961, in HTTPHandler.post(self, url, data, json, params, headers, stream, timeout, files, content, logging_obj)
    960 response = self.client.send(req, stream=stream)
--> 961 response.raise_for_status()
    962 return response

File ~/workspace/ws.study/ai-engineering/.venv/lib/python3.13/site-packages/httpx/_models.py:829, in Response.raise_for_status(self)
    828 message = message.format(self, error_type=error_type)
--> 829 raise HTTPStatusError(message, request=request, response=self)

HTTPStatusError: Client error '429 Too Many Requests' for url 'https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key=AIzaSyBivm0nktmWR-3dJQeT58c2GdpkikN0-1E'
For more information check: https://developer.mozilla.org/en-US/docs/Web/HTTP/Status/429

During handling of the above exception, another exception occurred:

VertexAIError                             Traceback (most recent call last)
File ~/workspace/ws.study/ai-engineering/.venv/lib/python3.13/site-packages/litellm/main.py:3113, in completion(model, messages, timeout, temperature, top_p, n, stream, stream_options, stop, max_completion_tokens, max_tokens, modalities, prediction, audio, presence_penalty, frequency_penalty, logit_bias, user, reasoning_effort, verbosity, response_format, seed, tools, tool_choice, logprobs, top_logprobs, parallel_tool_calls, web_search_options, deployment_id, extra_headers, safety_identifier, service_tier, functions, function_call, base_url, api_version, api_key, model_list, thinking, shared_session, **kwargs)
   3112     new_params = safe_deep_copy(optional_params or {})
-> 3113     response = vertex_chat_completion.completion(  # type: ignore
   3114         model=model,
   3115         messages=messages,
   3116         model_response=model_response,
   3117         print_verbose=print_verbose,
   3118         optional_params=new_params,
   3119         litellm_params=litellm_params,  # type: ignore
   3120         logger_fn=logger_fn,
   3121         encoding=_get_encoding(),
   3122         vertex_location=vertex_ai_location,
   3123         vertex_project=vertex_ai_project,
   3124         vertex_credentials=vertex_credentials,
   3125         gemini_api_key=gemini_api_key,
   3126         logging_obj=logging,
   3127         acompletion=acompletion,
   3128         timeout=timeout,
   3129         custom_llm_provider=custom_llm_provider,  # type: ignore
   3130         client=client,
... (출력 189줄 생략)
```

</div>


## 4. 프롬프트 캐싱

긴 프롬프트를 반복 사용할 때 비용을 절감할 수 있는 기법입니다.

### Prompt Caching with OpenAI

For OpenAI:

https://platform.openai.com/docs/guides/prompt-caching

> Cache hits are only possible for exact prefix matches within a prompt. To realize caching benefits, place static content like instructions and examples at the beginning of your prompt, and put variable content, such as user-specific information, at the end. This also applies to images and tools, which must be identical between requests.


Cached input is 4X cheaper

https://openai.com/api/pricing/


### Prompt Caching with Anthropic

https://docs.anthropic.com/en/docs/build-with-claude/prompt-caching

You have to tell Claude what you are caching

You pay 25% MORE to "prime" the cache

Then you pay 10X less to reuse from the cache with inputs.

https://www.anthropic.com/pricing#api

### Gemini supports both 'implicit' and 'explicit' prompt caching

https://ai.google.dev/gemini-api/docs/caching?lang=python


아래 예제에서는 셰익스피어의 햄릿 전문(약 4만 토큰)을 사용하여 실제 캐싱 효과를 확인합니다. 잘 되나요?


```python
# 햄릿 전문 로드 (약 4만 토큰)
with open("../../hamlet.txt", "r", encoding="utf-8") as f:
    hamlet_text = f.read()

print(f"햄릿 텍스트 길이: {len(hamlet_text):,} 문자")

# 첫 번째 호출 (캐시 프라이밍)
messages1 = [{"role": "user", "content": f"""다음은 셰익스피어의 햄릿 전문입니다:

{hamlet_text}

질문: 햄릿의 유명한 독백 "To be, or not to be"는 몇 막 몇 장에 등장하나요?"""}]

response1 = completion(model="openai/gpt-4o-mini", messages=messages1)

print("=== 첫 번째 호출 (캐시 프라이밍) ===")
print(f"입력 토큰: {response1.usage.prompt_tokens:,}")
if hasattr(response1.usage, 'prompt_tokens_details') and response1.usage.prompt_tokens_details:
    cached = getattr(response1.usage.prompt_tokens_details, 'cached_tokens', 0)
    print(f"캐시된 토큰: {cached:,}")
print(f"\n응답: {response1.choices[0].message.content}")
```



<div class="nb-output">

```text
햄릿 텍스트 길이: 191,726 문자
=== 첫 번째 호출 (캐시 프라이밍) ===
입력 토큰: 49,703
캐시된 토큰: 0

응답: 햄릿의 유명한 독백 "To be, or not to be"는 3막 1장에 등장합니다.
```

</div>



```python
# 두 번째 호출 (캐시 히트 기대)
messages2 = [{"role": "user", "content": f"""다음은 셰익스피어의 햄릿 전문입니다:

{hamlet_text}

질문: 오필리아는 어떻게 죽었나요?"""}]

response2 = completion(model="openai/gpt-4o-mini", messages=messages2)

print("=== 두 번째 호출 (캐시 히트) ===")
print(f"입력 토큰: {response2.usage.prompt_tokens:,}")

# 캐시 정보 확인
if hasattr(response2.usage, 'prompt_tokens_details') and response2.usage.prompt_tokens_details:
    cached = getattr(response2.usage.prompt_tokens_details, 'cached_tokens', 0)
    print(f"캐시된 토큰: {cached:,}")
    if cached > 0:
        cache_ratio = cached / response2.usage.prompt_tokens * 100
        print(f"캐시 히트율: {cache_ratio:.1f}%")
        print(f"💰 캐시된 토큰은 할인 적용!")

print(f"\n응답: {response2.choices[0].message.content}")
```



<div class="nb-output">

```text
=== 두 번째 호출 (캐시 히트) ===
입력 토큰: 49,685
캐시된 토큰: 49,536
캐시 히트율: 99.7%
💰 캐시된 토큰은 할인 적용!

응답: 오필리아는 "햄릿"에서 물에 빠져 죽은 것으로 묘사됩니다. 그녀는 괴로움과 슬픔에 압도되어 감정적으로 불안정한 상태에 있었고, 이는 결국 그녀의 죽음으로 이어집니다. 그녀가 물에 빠진 장소는 '버드나무가 시냇물 위로 기울어지는 곳'이라고 묘사되며, 그녀는 물속에서 꽃다발을 만들고 노래를 부르다가 갑자기 빠지게 됩니다. 어머니인 여왕이 그녀의 죽음을 듣고 슬퍼하는 장면이 등장합니다. 오필리아는 자신의 아버지인 폴로니우스를 잃은 슬픔과 삶의 압박감에 시달린 결과로, 비극적인 죽음을 맞이하게 됩니다.
```

</div>


---

## 5. 다중 에이전트 시스템

서로 다른 성격의 AI 에이전트들이 대화하는 시스템을 구현합니다.


```python
# 에이전트 정의
AGENTS = {
    "optimist": {
        "name": "희망이",
        "emoji": "😊",
        "system": """당신은 '희망이'입니다. 매우 긍정적이고 낙관적인 성격입니다.
        모든 상황에서 좋은 면을 찾으려 하고, 다른 사람들을 격려합니다.
        답변은 2-3문장으로 짧게 해주세요."""
    },
    "skeptic": {
        "name": "의심이",
        "emoji": "🤨",
        "system": """당신은 '의심이'입니다. 비판적 사고를 중시하는 회의론자입니다.
        주장에 대해 근거를 요구하고, 논리적 허점을 지적합니다. 하지만 공격적이지는 않습니다.
        답변은 2-3문장으로 짧게 해주세요."""
    },
    "mediator": {
        "name": "중재자",
        "emoji": "🤝",
        "system": """당신은 '중재자'입니다. 서로 다른 의견 사이에서 균형을 찾습니다.
        양쪽의 장점을 인정하고, 건설적인 결론을 도출하려 합니다.
        답변은 2-3문장으로 짧게 해주세요."""
    }
}
```



```python
def get_agent_response(agent_key: str, conversation: str, topic: str) -> str:
    """특정 에이전트의 응답을 생성합니다."""
    agent = AGENTS[agent_key]
    
    user_prompt = f"""현재 토론 주제: {topic}

지금까지의 대화:
{conversation}

당신({agent['name']})의 차례입니다. 위 대화에 이어서 의견을 말씀해주세요."""
    
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": agent["system"]},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0.8
    )
    
    return response.choices[0].message.content
```



```python
def run_discussion(topic: str, rounds: int = 4):
    """다중 에이전트 토론을 실행합니다."""
    conversation = "[토론 시작]\n"
    agent_order = ["optimist", "skeptic", "mediator"]
    
    print(f"📢 토론 주제: {topic}")
    print("=" * 50)
    
    for round_num in range(rounds):
        print(f"\n--- 라운드 {round_num + 1} ---")
        
        for agent_key in agent_order:
            agent = AGENTS[agent_key]
            response = get_agent_response(agent_key, conversation, topic)
            
            conversation += f"\n{agent['name']}: {response}"
            print(f"\n{agent['emoji']} {agent['name']}: {response}")
    
    return conversation
```



```python
# 토론 실행
topic = "AI가 인간의 창의성을 대체할 수 있을까?"
final_conversation = run_discussion(topic, rounds=3)
```



<div class="nb-output">

```text
📢 토론 주제: AI가 인간의 창의성을 대체할 수 있을까?
==================================================

--- 라운드 1 ---

😊 희망이: AI는 정말 멋진 도구로, 인간의 창의성을 보완하고 영감을 줄 수 있어요! 우리는 AI와 함께 협력함으로써 더 놀라운 아이디어를 만들어낼 수 있습니다. 창의성의 본질은 인간의 독창성과 감정에서 나오니, 걱정할 필요 없어요!

🤨 의심이: 희망이님의 주장에는 AI가 인간의 창의성을 "보완"할 수 있다는 점이 언급되었지만, AI가 어떻게 구체적으로 이 과정을 지원할 수 있는지에 대한 구체적인 예가 필요합니다. 또한, AI가 창의성을 대체할 가능성에 대한 우려를 간과하고 있는 것 같습니다. AI가 인간의 독창성과 감정을 이해할 수 있다는 보장이 있나요?

🤝 중재자: 희망이님은 AI가 창의성을 보완할 수 있는 가능성을 잘 강조하셨고, 이는 인간과 AI의 협업을 통해 새로운 아이디어를 창출할 수 있음을 시사합니다. 반면, 의심님은 AI의 한계와 대체 가능성에 대한 우려를 제기하며 더욱 구체적인 논의가 필요하다는 점을 강조하셨습니다. 두 의견 모두 중요한 시각을 제공하므로, AI의 역할과 한계를 명확하게 이해하는 것이 창의성의 미래를 논의하는 데 필요합니다.

--- 라운드 2 ---

😊 희망이: 희망이: 맞아요, 구체적인 예가 필요해요! 예를 들어, AI는 디자인이나 음악 작곡에서 다양한 스타일을 제안해줘서 인간이 그 아이디어를 바탕으로 더 깊이 있는 창작을 할 수 있게 도와줄 수 있어요. 우리는 협력하여 서로의 강점을 살릴 수 있는 미래를 만들어갈 수 있습니다!

🤨 의심이: 희망이님의 예시에서 AI가 디자인이나 음악 작곡에 기여할 수 있다고 언급하셨지만, AI가 제안하는 스타일이나 아이디어가 정말로 혁신적이고 독창적인지에 대한 의문이 남습니다. AI는 기존 데이터를 기반으로 작동하므로, 새로운 창의적 발상을 어떻게 만들어낼 수 있는지 구체적인 메커니즘이 필요합니다. 그리고 인간의 감정과 경험을 대체할 수 있는지에 대한 논의도 여전히 필요합니다.

🤝 중재자: 희망이님은 AI가 인간의 창의성을 보완하는 구체적인 예를 제시하며 협업의 가능성을 강조하셨습니다. 의심님은 AI의 혁신성과 독창성에 대한 우려를 제기하며 더 깊이 있는 논의의 필요성을 언급하셨습니다. 두 의견을 종합하면, AI의 기여를 이해하기 위해서는 실제 사례와 더불어 AI의 한계와 이를 극복할 방안에 대한 심도 있는 논의가 필요합니다.

--- 라운드 3 ---

😊 희망이: 희망이: 의심이님, 좋은 지적이에요! AI가 기존 데이터를 기반으로 하지만, 그 데이터를 활용해 새로운 조합과 시너지를 창출할 수 있습니다. 또한, 인간의 감정과 경험은 AI가 완전히 이해하지 못하더라도, 우리는 서로의 다름을 통해 더욱 풍부한 창작을 할 수 있어요. 결국 협력의 힘이 정말 중요하답니다!

🤨 의심이: 희망이님의 주장은 AI가 기존 데이터를 활용해 새로운 조합을 만든다고 하셨지만, 이러한 조합이 과연 '창의적'이라고 할 수 있는지에 대한 명확한 기준이 필요합니다. 또한, 협력의 힘이 중요하다는 점은 동의하지만, 인간의 감정과 경험을 완전히 이해하지 못하는 AI와의 협력이 실제로 얼마나 효과적일지에 대한 구체적인 논의가 부족합니다.

🤝 중재자: 희망이님은 AI가 기존 데이터를 활용해 새로운 조합을 만들어낼 수 있는 가능성과 협력의 중요성을 강조하셨습니다. 의심님은 이러한 조합이 진정한 창의성으로 인정받을 수 있는 기준과 AI와의 협력의 효과성에 대한 논의가 필요하다고 지적하셨습니다. 두 의견을 종합하면, AI의 창의적 기여를 평가하기 위한 명확한 기준 설정과 인간의 감정을 이해하는 방법에 대한 깊이 있는 논의가 필요할 것입니다.
```

</div>


---

## 6. LangChain 맛보기

LangChain은 LLM 애플리케이션 개발을 위한 프레임워크입니다.


```python
from langchain_openai import ChatOpenAI

# LangChain을 통한 모델 호출
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)

messages = [
    {"role": "user", "content": "머신러닝과 딥러닝의 차이를 한 문장으로 설명해주세요."}
]

response = llm.invoke(messages)
print("=== LangChain을 통한 GPT-4o-mini ===")
display(Markdown(response.content))
```



```python
# LangChain 체인 예시
from langchain_core.prompts import ChatPromptTemplate

# 프롬프트 템플릿
prompt = ChatPromptTemplate.from_messages([
    ("system", "당신은 {topic} 전문가입니다. 초보자에게 친절하게 설명해주세요."),
    ("user", "{question}")
])

# 체인 구성
chain = prompt | llm

# 체인 실행
response = chain.invoke({"topic": "Python", "question": "데코레이터가 뭔가요?"})
print("=== LangChain 체인 ===")
display(Markdown(response.content))
```



<div class="nb-output">

```text
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
Cell In[51], line 11
      5 prompt = ChatPromptTemplate.from_messages([
      6     ("system", "당신은 {topic} 전문가입니다. 초보자에게 친절하게 설명해주세요."),
      7     ("user", "{question}")
      8 ])
     10 # 체인 구성
---> 11 chain = prompt | llm
     13 # 체인 실행
     14 response = chain.invoke({"topic": "Python", "question": "데코레이터가 뭔가요?"})

NameError: name 'llm' is not defined
```

</div>


---

## 7. 로컬 LLM (Ollama) 심화

Ollama로 로컬에서 다양한 오픈소스 모델을 실행할 수 있습니다.


```python
import requests

# Ollama 서버 상태 확인
try:
    response = requests.get("http://localhost:11434/", timeout=5)
    print("✅ Ollama 서버가 실행 중입니다.")
    
    # 설치된 모델 목록
    tags_response = requests.get("http://localhost:11434/api/tags")
    if tags_response.status_code == 200:
        models = tags_response.json().get("models", [])
        print(f"\n📦 설치된 모델 ({len(models)}개):")
        for model in models[:5]:  # 상위 5개만 표시
            size_gb = model.get("size", 0) / (1024**3)
            print(f"   - {model['name']} ({size_gb:.1f}GB)")
except requests.exceptions.ConnectionError:
    print("❌ Ollama 서버가 실행되지 않았습니다.")
    print("   터미널에서 'ollama serve' 명령을 실행하세요.")
```



<div class="nb-output">

```text
✅ Ollama 서버가 실행 중입니다.

📦 설치된 모델 (3개):
   - exaone3.5:latest (4.4GB)
   - llama3.2:latest (1.9GB)
   - gpt-oss:latest (12.8GB)
```

</div>



```python
# Ollama 모델 호출 (OpenAI 호환 인터페이스)
ollama_client = OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="ollama"
)

try:
    response = ollama_client.chat.completions.create(
        model="exaone3.5",
        messages=[{"role": "user", "content": "What is Python? One sentence."}]
    )
    print("=== Ollama (Llama 3.2) ===")
    print(response.choices[0].message.content)
except Exception as e:
    print(f"오류: {e}")
```



<div class="nb-output">

```text
=== Ollama (Llama 3.2) ===
Python is a high-level programming language known for its readability and versatility, widely used for web development, data analysis, artificial intelligence, and more.
```

</div>


---

## 8. 실습: 3개 LLM 토론

OpenAI, Claude, Ollama 세 가지 LLM이 토론하는 시스템을 구현합니다.


```python
import anthropic

# 클라이언트 초기화
openai_client = OpenAI()
claude_client = anthropic.Anthropic()
ollama_client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")

# 시스템 프롬프트
PROMPTS = {
    "openai": "You are OpenAI's representative. You tend to be optimistic about AI. Keep responses to 2-3 sentences.",
    "claude": "You are Anthropic's representative. You emphasize AI safety. Keep responses to 2-3 sentences.",
    "ollama": "You are an open-source advocate. You value transparency. Keep responses to 2-3 sentences."
}

def get_openai_response(conversation: str, topic: str) -> str:
    response = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": PROMPTS["openai"]},
            {"role": "user", "content": f"Topic: {topic}\n\nConversation:\n{conversation}\n\nYour turn:"}
        ]
    )
    return response.choices[0].message.content

def get_claude_response(conversation: str, topic: str) -> str:
    response = claude_client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=200,
        system=PROMPTS["claude"],
        messages=[{"role": "user", "content": f"Topic: {topic}\n\nConversation:\n{conversation}\n\nYour turn:"}]
    )
    return response.content[0].text

def get_ollama_response(conversation: str, topic: str) -> str:
    try:
        response = ollama_client.chat.completions.create(
            model="llama3.2",
            messages=[
                {"role": "system", "content": PROMPTS["ollama"]},
                {"role": "user", "content": f"Topic: {topic}\n\nConversation:\n{conversation}\n\nYour turn:"}
            ]
        )
        return response.choices[0].message.content
    except:
        return "(Ollama not available)"
```



```python
# 3개 LLM 토론 실행
topic = "The future of open-source AI models"
conversation = ""

print(f"📢 Topic: {topic}")
print("=" * 50)

for round_num in range(2):
    print(f"\n--- Round {round_num + 1} ---")
    
    # OpenAI
    openai_reply = get_openai_response(conversation, topic)
    conversation += f"\nOpenAI: {openai_reply}"
    print(f"\n🟢 OpenAI: {openai_reply}")
    
    # Claude
    claude_reply = get_claude_response(conversation, topic)
    conversation += f"\nClaude: {claude_reply}"
    print(f"\n🟠 Claude: {claude_reply}")
    
    # Ollama
    ollama_reply = get_ollama_response(conversation, topic)
    conversation += f"\nOllama: {ollama_reply}"
    print(f"\n🔵 Ollama: {ollama_reply}")
```


---

## 9. 요약 및 다음 단계

### 이번 시리즈에서 학습한 내용

| Part | 주요 내용 |
|------|----------|
| **Part 1** | API 소개, 환경설정, 메시지 구조, 기본 호출, 활용 예시 |
| **Part 2** | 파라미터, 스트리밍, 에러처리, 다중 LLM, 비용 계산 |
| **Part 3** | 대화 이력, 캐싱, LiteLLM, 다중 에이전트, LangChain |

### 다음 단계로 배울 내용

| 주제 | 설명 |
|------|------|
| **Function Calling** | LLM이 외부 도구/API를 호출하는 방법 |
| **RAG** | 검색 증강 생성으로 최신 정보 활용 |
| **Agent** | 자율적으로 작업을 수행하는 AI 에이전트 |
| **Fine-tuning** | 특정 도메인에 맞게 모델 미세 조정 |
| **Prompt Engineering** | 더 효과적인 프롬프트 작성 기법 |

### 연습 문제

1. `ChatSession` 클래스에 토큰 사용량 추적 및 비용 계산 기능을 추가해보세요.
2. 다중 에이전트 토론에 4번째 에이전트(팩트 체커)를 추가해보세요.
3. LiteLLM을 사용하여 여러 모델의 응답 시간과 비용을 비교하는 벤치마크를 작성해보세요.
