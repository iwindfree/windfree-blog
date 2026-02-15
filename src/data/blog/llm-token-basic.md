---
title: "LLM 토큰 기본  개념"
author: iwindfree
pubDatetime: 2025-01-06T09:00:00Z
slug: "llm-token-basic"
category: "AI Engineering"
tags: ["ai", "llm", "tokenization"]
description: "이번 노트북에서는 LLM의 핵심 개념인 토큰Token 과 무상태Stateless 특성에 대해 알아봅니다."
---

이번 노트북에서는 LLM의 핵심 개념인 **토큰(Token)** 과 **무상태(Stateless)** 특성에 대해 알아봅니다.

## 개요

| 주제 | 내용 |
|------|------|
| 토큰(Token) | LLM이 텍스트를 처리하는 기본 단위 |
| 토큰화(Tokenization) | 텍스트를 토큰으로 분리하는 과정 |
| 한글 vs 영어 | 언어별 토큰 수 차이와 비용 영향 |
| Stateless | LLM API는 상태를 저장하지 않음 |
| 메모리 전략 | 대화 맥락을 유지하는 방법 |

## 학습 목표

1. 토큰의 개념과 토큰화 과정 이해하기
2. 한글과 영어의 토큰 수 차이 확인하기
3. LLM의 무상태(Stateless) 특성 이해하기
4. 대화 맥락을 유지하는 방법 알아보기

---

## 1. 언어 모델(Language Model)이란?

언어 모델 (Language Model) 은 텍스트 데이터의 통계 정보를 기반으로 자연어를 이해하고 생성하는 기능을 제공합니다. 

### 핵심 원리
언어모델의 기본 아이디어는 다음에 올 토큰 을 예측하는 것입니다. 예를 들어 '오늘 날씨가 정말 __' 이라는 문장이 주어지면, 모델은 '좋다', '덥다' 같은 단어가 올 확률을 계산합니다. 이 단순한 원리가 충분히 큰 규모로 학습되면 놀라운 기능을 제공하게 됩니다.

### 발전과정
#### 통계적 언어모델 (n-gram)
초기에는 이전 n개 단어만 보고 다음 단어를 예측했습니다. 단순하지만 긴 문맥을 이해하지 못하는 한계가 있었습니다.

#### 신경망 기반 언어모델 (RNN/LSTM/GRU 등)
단어를 벡터로 표현하고 순환신경망을 통해 확률을 계산하는 방식입니다. 순차적으로 정보를 처리하며, 더 긴 문맥을 기억할 수 있게 되었지만, 긴 문장 학습이 어려웠고 병렬 처리가 어려워서 학습 속도가 느리다는 단점이 존재합니다.

#### Transformer (2017)
"Attention is All You need" 라는 논문에서 핵심 아이디어가 도출되었습니다. Attention 매커니즘을 도입해 문장 내 모든 단어간의 관계를 병렬로 처리하여 맥락 문맥 이해가 가능하게 되었고 학습 속도가 급상승하게 되었습니다. 성능의 비약적 향상으로 LLM 의 기반 아키텍처가 됩니다. 트랜스포머는 현재 ChatGPT 같은 생성형 AI의 근간이 되는 모델 구조입니다. 이전 모델들과의 가장 큰 차이점은 문장을 한 단어씩 순서대로 읽지 않고, 문장 전체를 한꺼번에 보고 단어 사이의 관계를 파악한다는 점이에요.

##### 주요 구성 요소
어텐션 (Attention) 메커니즘: 문장 안에서 어떤 단어가 중요한지 '주목'하는 기술입니다.

예: "그는 사과를 먹으려 했지만 그것이 너무 딱딱했다."

여기서 '그것'이 '사과'를 가리킨다는 것을 어텐션이 찾아내어 연결해 줍니다.

##### 인코더(Encoder)와 디코더(Decoder):

인코더: 입력된 문장의 의미를 깊게 이해하고 요약합니다.

디코더: 인코더의 정보를 바탕으로 적절한 답변이나 다음 단어를 생성합니다. 현재 대부분의 LLM 이 디코더 방식을 많이 사용하고 있습니다. 

---

## 2. 토큰(Token)

언어 모델의 기반 단위는 토큰입니다. 토큰은 모델에 따라 문자, 단어, 또는 단어의 일부가 될 수 있으며, GPT-4 인 경우는 토큰 하나의 평균 길이는 단어의 3/4 정도라고 말하고 있습니다.

### 토크나이저 라이브러리

각 LLM은 고유한 토크나이저를 사용합니다. 같은 텍스트라도 모델에 따라 토큰 수가 다를 수 있습니다.

| LLM | 토크나이저 | 설명 |
|-----|-----------|------|
| OpenAI (GPT) | **tiktoken** | OpenAI 공식 라이브러리 |
| Meta (LLaMA) | SentencePiece | Google에서 개발한 오픈소스 |
| Google (Gemma) | SentencePiece | BPE/Unigram 알고리즘 지원 |
| Anthropic (Claude) | 자체 토크나이저 | 비공개 |
| Hugging Face 모델들 | `transformers.AutoTokenizer` | 모델별 자동 로드 |

### tiktoken 라이브러리

이번 교육에서는 OpenAI 모델을 주로 사용하므로 **tiktoken**을 사용합니다. tiktoken은 OpenAI에서 제공하는 토크나이저 라이브러리로, GPT 모델들이 실제로 사용하는 토큰화 방식을 Python에서 직접 확인할 수 있습니다.

| 기능 | 설명 |
|------|------|
| `encoding_for_model()` | 특정 모델의 인코딩 방식 가져오기 |
| `encode()` | 텍스트 → 토큰 ID 리스트 변환 |
| `decode()` | 토큰 ID → 텍스트 변환 |

tiktoken을 사용하면 API 호출 전에 토큰 수를 미리 계산하여 비용을 예측할 수 있습니다.

간단히 파이썬 코드를 통해서 개념을 살펴보겠습니다.


```python
import tiktoken
encoding = tiktoken.encoding_for_model("gpt-4")
text="hello, my name is windfree."
tokens = encoding.encode(text)
print(tokens)
```



<div class="nb-output">

```text
[15339, 11, 856, 836, 374, 10160, 10816, 13]
```

</div>



```python
for token_id in tokens:
    token_text = encoding.decode([token_id])
    print(f"Token ID: {token_id}, Token Text:{token_text}")
```



<div class="nb-output">

```text
Token ID: 15339, Token Text:hello
Token ID: 11, Token Text:,
Token ID: 856, Token Text: my
Token ID: 836, Token Text: name
Token ID: 374, Token Text: is
Token ID: 10160, Token Text: wind
Token ID: 10816, Token Text:free
Token ID: 13, Token Text:.
```

</div>


토큰을 확인해 보았을 때 앞에 공백이 포함되는 경우를 확인할 수 있습니다. 이것은 위치 정보까지 토큰에 포함하기 위해서 입니다. **토큰화** 는 원문을 모델이 정한 길이로 나누는 과정을 말하는 것이며 모델이 다룰 수 있는 토큰의 집합을 모델의 **어휘** 라고 부릅니다. 소수의 토큰을 사용해 많은 단어를 만들 수 있으며 GPT-4 의 어휘 크기는 100,256 개로 알려져 있습니다.

### 한글과 영어의 토큰 수 비교

같은 의미의 문장이라도 언어에 따라 토큰 수가 크게 다릅니다. 한글은 영어보다 더 많은 토큰을 사용하는 경향이 있어 API 비용에 영향을 줍니다.


```python
# 영어와 한글 토큰 수 비교
def compare_tokens(text_en, text_ko):
    tokens_en = encoding.encode(text_en)
    tokens_ko = encoding.encode(text_ko)
    print(f"영어: \"{text_en}\"")
    print(f"  → 토큰 수: {len(tokens_en)}")
    print(f"한글: \"{text_ko}\"")
    print(f"  → 토큰 수: {len(tokens_ko)}")
    print(f"  → 한글이 {len(tokens_ko) / len(tokens_en):.1f}배 더 많은 토큰 사용\n")

compare_tokens("Hello, how are you?", "안녕하세요, 어떻게 지내세요?")
compare_tokens("The weather is nice today.", "오늘 날씨가 좋네요.")
compare_tokens("I love programming.", "나는 프로그래밍을 좋아합니다.")
```



<div class="nb-output">

```text
영어: "Hello, how are you?"
  → 토큰 수: 6
한글: "안녕하세요, 어떻게 지내세요?"
  → 토큰 수: 16
  → 한글이 2.7배 더 많은 토큰 사용

영어: "The weather is nice today."
  → 토큰 수: 6
한글: "오늘 날씨가 좋네요."
  → 토큰 수: 15
  → 한글이 2.5배 더 많은 토큰 사용

영어: "I love programming."
  → 토큰 수: 4
한글: "나는 프로그래밍을 좋아합니다."
  → 토큰 수: 13
  → 한글이 3.2배 더 많은 토큰 사용
```

</div>


### LLM 은 stateless 합니다.
아래의 코드를 수행해보면 재미있는 현상을 발견할 수 있습니다.


```python
import os
from dotenv import load_dotenv

load_dotenv(override=True)
api_key = os.getenv('OPENAI_API_KEY')
if not api_key:
    print("No API key was found")
else:
    print("API key found.")
```



<div class="nb-output">

```text
API key found.
```

</div>



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
As an artificial intelligence, I don't have access to personal data about individuals unless it has been shared with me in the course of our conversation. I am designed to respect user privacy and confidentiality. Therefore, I don't know your name.
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


당연한 얘기일 수 있지만, 다시 한번 정리해보면:

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

### Sliding Window 전략

대화가 길어지면 컨텍스트 윈도우 한계에 도달하거나 비용이 급증합니다. Sliding Window는 최근 N개의 메시지만 유지하고 오래된 메시지는 삭제하는 전략입니다.


```python
def sliding_window(messages, max_messages=10):
    """최근 N개의 메시지만 유지하는 Sliding Window 전략"""
    # system 메시지는 항상 유지
    system_messages = [m for m in messages if m["role"] == "system"]
    other_messages = [m for m in messages if m["role"] != "system"]
    
    # 최근 메시지만 유지
    if len(other_messages) > max_messages:
        other_messages = other_messages[-max_messages:]
    
    return system_messages + other_messages

# 예시: 긴 대화 시뮬레이션
long_conversation = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Message 1"},
    {"role": "assistant", "content": "Response 1"},
    {"role": "user", "content": "Message 2"},
    {"role": "assistant", "content": "Response 2"},
    {"role": "user", "content": "Message 3"},
    {"role": "assistant", "content": "Response 3"},
    {"role": "user", "content": "Message 4"},
    {"role": "assistant", "content": "Response 4"},
    {"role": "user", "content": "Message 5"},
    {"role": "assistant", "content": "Response 5"},
    {"role": "user", "content": "What was my first message?"},  # 가장 최근 질문
]

print(f"원본 메시지 수: {len(long_conversation)}")
trimmed = sliding_window(long_conversation, max_messages=6)
print(f"Sliding Window 적용 후: {len(trimmed)}")
print("\n유지된 메시지:")
for msg in trimmed:
    print(f"  [{msg['role']}] {msg['content'][:30]}...")
```



<div class="nb-output">

```text
원본 메시지 수: 12
Sliding Window 적용 후: 7

유지된 메시지:
  [system] You are a helpful assistant....
  [assistant] Response 3...
  [user] Message 4...
  [assistant] Response 4...
  [user] Message 5...
  [assistant] Response 5...
  [user] What was my first message?...
```

</div>


---

## 요약

이번 노트북에서는 LLM의 핵심 개념인 토큰과 메모리에 대해 알아보았습니다.

### 핵심 포인트

1. **토큰**: LLM이 처리하는 기본 단위. 단어, 부분 단어, 또는 문자가 될 수 있음
2. **언어별 토큰 차이**: 한글은 영어보다 약 2배 이상의 토큰을 사용
3. **Stateless**: LLM API 호출은 무상태. 매번 전체 대화를 전송해야 맥락 유지
4. **비용 구조**: 입력/출력 토큰에 따라 과금, 대화가 길어질수록 비용 증가
5. **메모리 전략**: Sliding Window, 요약, RAG 등으로 컨텍스트 관리

### 다음 단계

다음 노트북에서는 실제 LLM API를 호출하는 방법과 다양한 파라미터 활용법을 알아봅니다.
