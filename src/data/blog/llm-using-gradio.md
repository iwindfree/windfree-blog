---
title: "Gradio 기본 사용법"
author: iwindfree
pubDatetime: 2025-01-27T09:00:00Z
slug: "llm-using-gradio"
category: "AI Engineering"
tags: ["ai", "llm", "gradio"]
description: "Gradiohttps://gradio.app는 머신러닝 모델이나 Python 함수를 위한 웹 UI를 빠르게 만들 수 있는 프레임워크입니다."
---

## Gradio란?

[Gradio](https://gradio.app)는 머신러닝 모델이나 Python 함수를 위한 웹 UI를 빠르게 만들 수 있는 프레임워크입니다.

### 주요 특징

| 특징 | 설명 |
|------|------|
| 빠른 프로토타이핑 | 몇 줄의 코드로 데모 UI 생성 |
| 공유 가능 | `share=True`로 임시 공개 URL 생성 |
| 다양한 컴포넌트 | 텍스트, 이미지, 오디오, 비디오 등 지원 |
| HuggingFace 통합 | Spaces에 쉽게 배포 가능 |

```bash
pip install gradio
```


```python
import gradio as gr
import os
from openai import OpenAI
```


---

## 2. 기본 컴포넌트

Gradio는 다양한 입출력 컴포넌트를 제공합니다.

### 주요 컴포넌트

| 컴포넌트 | 용도 |
|----------|------|
| `gr.Textbox` | 텍스트 입력/출력 |
| `gr.Number` | 숫자 입력 |
| `gr.Slider` | 범위 내 값 선택 |
| `gr.Checkbox` | 불리언 선택 |
| `gr.Dropdown` | 드롭다운 메뉴 |
| `gr.Button` | 클릭 이벤트 |
| `gr.Image` | 이미지 입출력 |


```python
# 가장 간단한 예제: 인사하기
def greet(name):
    return f"안녕하세요, {name}님!"

demo = gr.Interface(
    fn=greet,
    inputs=gr.Textbox(label="이름", placeholder="이름을 입력하세요"),
    outputs=gr.Textbox(label="인사말"),
    title="인사하기",
    description="이름을 입력하면 인사말을 생성합니다."
)

demo.launch()
```



<div class="nb-output">

```text
* Running on local URL:  http://127.0.0.1:7863
* To create a public link, set `share=True` in `launch()`.
<IPython.core.display.HTML object>
```

</div>



```python
# 여러 입력 컴포넌트 조합
def calculate_bmi(height, weight, unit):
    if unit == "cm/kg":
        height_m = height / 100
    else:
        height_m = height
    
    bmi = weight / (height_m ** 2)
    
    if bmi < 18.5:
        category = "저체중"
    elif bmi < 25:
        category = "정상"
    elif bmi < 30:
        category = "과체중"
    else:
        category = "비만"
    
    return f"BMI: {bmi:.1f} ({category})"

demo = gr.Interface(
    fn=calculate_bmi,
    inputs=[
        gr.Number(label="키", value=170),
        gr.Slider(minimum=30, maximum=150, value=70, label="몸무게 (kg)"),
        gr.Dropdown(["cm/kg", "m/kg"], value="cm/kg", label="단위")
    ],
    outputs=gr.Textbox(label="결과"),
    title="BMI 계산기"
)

demo.launch()
```



<div class="nb-output">

```text
* Running on local URL:  http://127.0.0.1:7862
* To create a public link, set `share=True` in `launch()`.
<IPython.core.display.HTML object>
```

</div>


---

## 3. Interface vs Blocks

Gradio는 두 가지 방식으로 UI를 구성할 수 있습니다.

### gr.Interface
- 간단한 입력 → 함수 → 출력 패턴
- 빠른 프로토타이핑에 적합
- 레이아웃 커스터마이징 제한적

### gr.Blocks
- 완전한 레이아웃 제어
- 복잡한 이벤트 처리 가능
- 여러 컴포넌트 간 상호작용
- 탭, 행, 열 등 레이아웃 구성

| 상황 | 추천 |
|------|------|
| 단순 데모, 빠른 테스트 | Interface |
| 복잡한 UI, 여러 버튼/이벤트 | Blocks |
| 채팅 UI | ChatInterface 또는 Blocks |


```python
# Blocks를 사용한 커스텀 레이아웃
def process_text(text, uppercase, add_emoji):
    result = text
    if uppercase:
        result = result.upper()
    if add_emoji:
        result = f"✨ {result} ✨"
    return result

with gr.Blocks(title="텍스트 변환기") as demo:
    gr.Markdown("# 텍스트 변환기")
    gr.Markdown("텍스트를 입력하고 옵션을 선택하세요.")
    
    with gr.Row():
        with gr.Column():
            input_text = gr.Textbox(label="입력", lines=3)
            with gr.Row():
                uppercase = gr.Checkbox(label="대문자 변환")
                add_emoji = gr.Checkbox(label="이모지 추가")
            btn = gr.Button("변환", variant="primary")
        
        with gr.Column():
            output_text = gr.Textbox(label="결과", lines=3)
    
    btn.click(fn=process_text, inputs=[input_text, uppercase, add_emoji], outputs=output_text)

demo.launch()
```



<div class="nb-output">

```text
* Running on local URL:  http://127.0.0.1:7863
* To create a public link, set `share=True` in `launch()`.
<IPython.core.display.HTML object>
```

</div>


---

## 4. LLM 연동 예제

Gradio는 LLM과 연동한 채팅 UI를 쉽게 만들 수 있습니다. `gr.ChatInterface`를 사용하면 몇 줄의 코드로 채팅 UI가 완성됩니다.

### ChatInterface의 콜백 함수

`gr.ChatInterface`에 전달하는 함수는 두 개의 파라미터를 받습니다:

```python
def chat(message, history):
    # message: 현재 사용자가 입력한 메시지 (str)
    # history: 이전 대화 기록 
    ...
```


이를 OpenAI API의 messages 형식으로 변환해야 합니다:


```python
import os
from dotenv import load_dotenv
from openai import OpenAI
# 환경변수 로드
load_dotenv(override=True)
```



<div class="nb-output">

```text
True
```

</div>



```python
# OpenAI API 연동 채팅
client = OpenAI()
system_message = "You are a helpful assistant"
MODEL = 'gpt-4.1-mini'
def chat(message, history):
    history = [{"role":h["role"], "content":h["content"]} for h in history]  # history를 OpenAI API 형식에 맞게 변환 (딕셔너리 리스트)
    messages = [{"role": "system", "content": system_message}] + history + [{"role": "user", "content": message}]
    response = client.chat.completions.create(model=MODEL, messages=messages)
    return response.choices[0].message.content

demo = gr.ChatInterface(
    fn=chat,
    title="GPT 챗봇",
    description="GPT-4o-mini와 대화해보세요."
)

demo.launch()
```



<div class="nb-output">

```text
* Running on local URL:  http://127.0.0.1:7867
* To create a public link, set `share=True` in `launch()`.
<IPython.core.display.HTML object>
```

</div>



```python
# 스트리밍 응답 구현
def chat_stream(message, history):
    system_message = "You are a helpful assistant"
    MODEL = 'gpt-4.1-mini'
    history = [{"role":h["role"], "content":h["content"]} for h in history]
    messages = [{"role": "system", "content": system_message}] + history + [{"role": "user", "content": message}]
    messages.append({"role": "user", "content": message})
    
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        stream=True  # 스트리밍 활성화
    )
    
    partial_message = ""
    for chunk in response:
        if chunk.choices[0].delta.content:
            partial_message += chunk.choices[0].delta.content
            yield partial_message  # 점진적으로 반환

demo = gr.ChatInterface(
    fn=chat_stream,
    title="GPT 챗봇 (스트리밍)",
    description="실시간으로 응답이 생성됩니다."
)

demo.launch()
```



<div class="nb-output">

```text
* Running on local URL:  http://127.0.0.1:7864
* To create a public link, set `share=True` in `launch()`.
<IPython.core.display.HTML object>
```

</div>


---

## 요약

이번 노트북에서는 Gradio의 기본 사용법을 알아보았습니다.

### 핵심 포인트

1. **Gradio**: ML/AI 데모를 위한 빠른 UI 프레임워크
2. **컴포넌트**: Textbox, Slider, Button 등 다양한 입출력 지원
3. **Interface**: 간단한 함수 래핑, 빠른 프로토타이핑
4. **Blocks**: 복잡한 레이아웃과 이벤트 처리
5. **ChatInterface**: LLM 채팅 UI를 쉽게 구현
