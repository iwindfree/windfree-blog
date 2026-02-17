---
title: "System Message 활용하기"
author: iwindfree
pubDatetime: 2025-01-20T09:00:00Z
slug: "llm-system-message"
category: "LLM Engineering"
series: "LLM Engineering"
seriesOrder: 7
tags: ["ai", "llm", "prompting"]
description: "System message를 활용하면 챗봇에 특정 역할, 컨텍스트, 행동 지침을 부여할 수 있습니다. 이번 노트북에서는 System message를 효과적으로 활용하는 방법을 알아봅니다."
---

System message를 활용하면 챗봇에 특정 역할, 컨텍스트, 행동 지침을 부여할 수 있습니다. 이번 노트북에서는 System message를 효과적으로 활용하는 방법을 알아봅니다.

## System Message란?

System message는 LLM에게 전달하는 **초기 지침**입니다. 대화가 시작되기 전에 AI의 역할, 성격, 제약사항 등을 설정합니다.

```python
messages = [
    {"role": "system", "content": "당신은 친절한 고객 상담사입니다."},
    {"role": "user", "content": "안녕하세요"},
    ...
]
```


```python
import gradio as gr
from openai import OpenAI

client = OpenAI()
```


---

## 1. 비즈니스 컨텍스트 제공

System message에 비즈니스 정보와 역할을 설정하면 일관된 응답을 유도할 수 있습니다.

### 효과적인 System Message 구성 요소

| 구성 요소 | 설명 | 예시 |
|----------|------|------|
| 역할 정의 | AI가 맡을 역할 | "당신은 레스토랑 예약 도우미입니다" |
| 컨텍스트 정보 | 필요한 배경 지식 | 영업시간, 메뉴, 위치 등 |
| 행동 지침 | 응대 방식 가이드 | "친절하게 응대하세요" |
| 제약사항 | 하지 말아야 할 것 | "가격 할인은 약속하지 마세요" |

---

## 2. One-shot Prompting

예시 답변을 system message에 포함시키면 LLM이 원하는 스타일로 응답하도록 유도할 수 있습니다.

```
예를 들어, 고객이 '예약하고 싶어요'라고 하면,
'네, 예약 도와드리겠습니다! 몇 분이서 방문하실 예정인가요?'처럼 응답하세요.
```

이렇게 하면 LLM이 비슷한 톤과 형식으로 응답합니다.


```python
# 식당 예약 도우미 예제
system_message = """당신은 이탈리안 레스토랑 '벨라 이탈리아'의 예약 도우미입니다.

[레스토랑 정보]
- 영업시간: 점심 11:30-14:30, 저녁 17:30-22:00 (월요일 휴무)
- 위치: 서울시 강남구 테헤란로 123
- 예약: 최소 2인부터, 최대 8인까지 가능
- 인기 메뉴: 트러플 파스타, 마르게리타 피자, 티라미수

[응대 지침]
- 친절하고 전문적인 톤을 유지하세요
- 예약 문의 시 날짜, 시간, 인원을 순서대로 확인하세요
- 메뉴 추천 요청 시 인기 메뉴를 안내하세요

예를 들어, 고객이 '예약하고 싶어요'라고 하면, 
'네, 예약 도와드리겠습니다! 몇 분이서 방문하실 예정인가요?'처럼 응답하세요."""

def restaurant_chat(message, history):
    messages = [{"role": "system", "content": system_message}]
    for user_msg, assistant_msg in history:
        messages.append({"role": "user", "content": user_msg})
        messages.append({"role": "assistant", "content": assistant_msg})
    messages.append({"role": "user", "content": message})
    
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        stream=True
    )
    
    partial_message = ""
    for chunk in response:
        if chunk.choices[0].delta.content:
            partial_message += chunk.choices[0].delta.content
            yield partial_message

demo = gr.ChatInterface(
    fn=restaurant_chat,
    title="벨라 이탈리아 예약 도우미",
    description="레스토랑 예약 및 메뉴 문의를 도와드립니다."
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


---

## 3. 동적 System Message

사용자 입력에 따라 system message를 동적으로 수정하면 상황에 맞는 응답을 유도할 수 있습니다.

### 활용 사례

- **키워드 감지**: 특정 단어가 포함되면 관련 정보 추가
- **사용자 상태 반영**: VIP 고객, 신규 고객 등에 따라 다른 응대
- **시간대별 안내**: 점심/저녁 시간에 따라 다른 메뉴 추천


```python
# 동적 System Message 예제
def restaurant_chat_dynamic(message, history):
    # 기본 system message
    dynamic_system_message = system_message
    
    # 특정 키워드에 따라 추가 지침 삽입
    if "할인" in message or "이벤트" in message:
        dynamic_system_message += "\n\n[현재 진행 중인 이벤트] 평일 런치 타임(11:30-14:30)에는 파스타 메뉴 20% 할인 중입니다. 이 정보를 안내해주세요."
    
    if "주차" in message:
        dynamic_system_message += "\n\n[주차 안내] 건물 지하 주차장 2시간 무료 주차 가능합니다. 이 정보를 안내해주세요."
    
    if "단체" in message or "회식" in message:
        dynamic_system_message += "\n\n[단체 예약] 6인 이상 단체 예약 시 별도 룸 이용 가능하며, 코스 메뉴를 추천해주세요."
    
    messages = [{"role": "system", "content": dynamic_system_message}]
    for user_msg, assistant_msg in history:
        messages.append({"role": "user", "content": user_msg})
        messages.append({"role": "assistant", "content": assistant_msg})
    messages.append({"role": "user", "content": message})
    
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        stream=True
    )
    
    partial_message = ""
    for chunk in response:
        if chunk.choices[0].delta.content:
            partial_message += chunk.choices[0].delta.content
            yield partial_message

demo = gr.ChatInterface(
    fn=restaurant_chat_dynamic,
    title="벨라 이탈리아 예약 도우미 (동적 프롬프트)",
    description="할인, 주차, 단체 예약 등을 물어보세요!"
)

demo.launch()
```



<div class="nb-output">

```text
* Running on local URL:  http://127.0.0.1:7861
* To create a public link, set `share=True` in `launch()`.
<IPython.core.display.HTML object>
```

</div>


---

## 요약

이번 노트북에서는 System Message를 효과적으로 활용하는 방법을 알아보았습니다.

### 핵심 포인트

1. **역할과 컨텍스트 설정**: 비즈니스 정보, 응대 지침을 system message에 포함
2. **One-shot Prompting**: 예시 답변을 포함하여 응답 스타일 유도
3. **동적 프롬프트**: 사용자 입력에 따라 system message 수정

### 팁

- System message는 **구조화**하면 관리하기 쉽습니다 (섹션 구분, 마크다운 활용)
- 너무 긴 system message는 토큰 비용을 증가시키므로 **핵심만** 포함하세요
- 동적 프롬프트는 **RAG(검색 증강 생성)**의 기초가 됩니다
