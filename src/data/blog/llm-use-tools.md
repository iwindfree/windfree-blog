---
title: "Tool Use (Function Calling)"
author: iwindfree
pubDatetime: 2025-02-03T09:00:00Z
slug: "llm-use-tools"
category: "AI Engineering"
tags: ["ai", "llm", "tool-use"]
description: "LLM에게 도구Tool 를 제공하면, AI가 외부 함수를 호출하여 실시간 정보를 가져오거나 작업을 수행할 수 있습니다."
---

LLM에게 **도구(Tool)** 를 제공하면, AI가 외부 함수를 호출하여 실시간 정보를 가져오거나 작업을 수행할 수 있습니다.

## Tool Use란?

Tool Use(또는 Function Calling)는 LLM이 직접 함수를 실행하는 것이 아니라, **"이 함수를 이런 인자로 호출해달라"** 고 요청하는 방식입니다.

```
사용자: "파이썬 코딩의 기술 책 가격이 얼마야?"
    ↓
LLM: "get_book_price('파이썬 코딩의 기술') 함수를 호출해주세요"
    ↓
우리 코드: 함수 실행 → 결과 반환
    ↓
LLM: "파이썬 코딩의 기술의 가격은 32,000원입니다."
```

### 활용 사례

| 분야 | Tool 예시 |
|------|----------|
| 고객 지원 | 주문 조회, 배송 추적, 환불 처리 |
| 정보 검색 | DB 조회, API 호출, 웹 검색 |
| 업무 자동화 | 이메일 발송, 캘린더 등록, 문서 생성 |

## Tool Use 동작 원리

**중요**: LLM은 Tool을 직접 실행하지 않습니다. LLM은 "이 함수를 호출해달라"고 **요청**만 하고, 실제 실행은 **애플리케이션**이 담당합니다.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                                애플리케이션                                  │
│  ┌───────────────────────────────────────────────────────────────────────┐  │
│  │                                                                       │  │
│  │   ┌─────────┐      1. 메시지 + tools 전송      ┌─────────────────┐   │  │
│  │   │         │ ─────────────────────────────▶  │                 │   │  │
│  │   │ 클라이언 │                                  │                 │   │  │
│  │   │ 트 코드 │      2. "이 함수 호출해줘"        │    LLM (API)    │   │  │
│  │   │         │ ◀─────────────────────────────  │                 │   │  │
│  │   │         │      (함수명 + 인자 반환)        │                 │   │  │
│  │   └────┬────┘                                  └─────────────────┘   │  │
│  │        │                                                              │  │
│  │        │ 3. 함수 직접 실행                                            │  │
│  │        ▼                                                              │  │
│  │   ┌─────────┐                                                         │  │
│  │   │  Tool   │  get_book_info("클린 코드")                             │  │
│  │   │  함수들  │  → "클린 코드 - 저자: 로버트 마틴, 가격: 33,000원"       │  │
│  │   └────┬────┘                                                         │  │
│  │        │                                                              │  │
│  │        │ 4. 실행 결과                                                 │  │
│  │        ▼                                                              │  │
│  │   ┌─────────┐      5. 결과 전송                ┌─────────────────┐   │  │
│  │   │ 클라이언 │ ─────────────────────────────▶  │                 │   │  │
│  │   │ 트 코드 │                                  │    LLM (API)    │   │  │
│  │   │         │      6. 최종 응답 생성            │                 │   │  │
│  │   │         │ ◀─────────────────────────────  │                 │   │  │
│  │   └─────────┘   "클린 코드는 33,000원입니다"   └─────────────────┘   │  │
│  │                                                                       │  │
│  └───────────────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 핵심 포인트

| 단계 | 주체 | 역할 |
|------|------|------|
| 1 | 클라이언트 | 사용자 메시지와 사용 가능한 tools 목록을 LLM에 전송 |
| 2 | **LLM** | 메시지를 분석하고, 필요한 함수와 인자를 **결정** (실행 X) |
| 3 | **클라이언트** | LLM이 요청한 함수를 **직접 실행** |
| 4 | 클라이언트 | 함수 실행 결과를 받음 |
| 5 | 클라이언트 | 실행 결과를 LLM에 다시 전송 |
| 6 | LLM | 결과를 바탕으로 사용자에게 보여줄 최종 응답 생성 |

> **왜 이렇게 설계했을까?**  
> - **보안**: LLM이 직접 코드를 실행하면 위험할 수 있음  
> - **제어**: 어떤 함수를 허용할지 개발자가 결정  
> - **유연성**: 어떤 언어, 어떤 시스템의 함수든 연결 가능


```python
import os
import json
from openai import OpenAI
import gradio as gr

client = OpenAI()
MODEL = "gpt-4o-mini"
```


---

## 예제: 온라인 서점 도우미

도서 정보를 조회하고, 재고를 확인하는 온라인 서점 AI 어시스턴트를 만들어봅니다.


```python
# 도서 데이터 (실제로는 DB나 API에서 가져옴)
books_db = {
    "파이썬 코딩의 기술": {
        "author": "브렛 슬라킨",
        "price": 32000,
        "stock": 15,
        "category": "프로그래밍"
    },
    "클린 코드": {
        "author": "로버트 마틴",
        "price": 33000,
        "stock": 8,
        "category": "프로그래밍"
    },
    "데이터 과학을 위한 통계": {
        "author": "피터 브루스",
        "price": 28000,
        "stock": 0,
        "category": "데이터 과학"
    },
    "딥러닝 입문": {
        "author": "사이토 고키",
        "price": 24000,
        "stock": 23,
        "category": "인공지능"
    }
}
```


---

## 1. Tool 함수 정의

LLM이 호출할 수 있는 함수들을 정의합니다.


```python
def get_book_info(title: str) -> str:
    """도서 정보를 조회합니다."""
    print(f"[Tool 호출] get_book_info('{title}')")
    
    book = books_db.get(title)
    if book:
        return f"'{title}' - 저자: {book['author']}, 가격: {book['price']:,}원, 카테고리: {book['category']}"
    return f"'{title}' 도서를 찾을 수 없습니다."

def check_stock(title: str) -> str:
    """도서 재고를 확인합니다."""
    print(f"[Tool 호출] check_stock('{title}')")
    
    book = books_db.get(title)
    if book:
        stock = book['stock']
        if stock > 0:
            return f"'{title}' 재고: {stock}권 (구매 가능)"
        return f"'{title}' 현재 품절입니다. 입고 예정일을 확인해주세요."
    return f"'{title}' 도서를 찾을 수 없습니다."

def search_by_category(category: str) -> str:
    """카테고리별 도서를 검색합니다."""
    print(f"[Tool 호출] search_by_category('{category}')")
    
    results = [title for title, info in books_db.items() if info['category'] == category]
    if results:
        return f"{category} 카테고리 도서: {', '.join(results)}"
    return f"{category} 카테고리에 해당하는 도서가 없습니다."
```



```python
# 함수 테스트
print(get_book_info("클린 코드"))
print(check_stock("데이터 과학을 위한 통계"))
print(search_by_category("프로그래밍"))
```



<div class="nb-output">

```text
[Tool 호출] get_book_info('클린 코드')
'클린 코드' - 저자: 로버트 마틴, 가격: 33,000원, 카테고리: 프로그래밍
[Tool 호출] check_stock('데이터 과학을 위한 통계')
'데이터 과학을 위한 통계' 현재 품절입니다. 입고 예정일을 확인해주세요.
[Tool 호출] search_by_category('프로그래밍')
프로그래밍 카테고리 도서: 파이썬 코딩의 기술, 클린 코드
```

</div>


---

## 2. Tool 스키마 정의

LLM에게 함수의 이름, 설명, 파라미터를 알려주는 스키마를 정의합니다. JSON Schema 형식을 사용합니다.


```python
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_book_info",
            "description": "도서의 상세 정보(저자, 가격, 카테고리)를 조회합니다.",
            "parameters": {
                "type": "object",
                "properties": {
                    "title": {
                        "type": "string",
                        "description": "조회할 도서의 제목"
                    }
                },
                "required": ["title"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "check_stock",
            "description": "도서의 재고 수량을 확인합니다.",
            "parameters": {
                "type": "object",
                "properties": {
                    "title": {
                        "type": "string",
                        "description": "재고를 확인할 도서의 제목"
                    }
                },
                "required": ["title"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "search_by_category",
            "description": "특정 카테고리의 도서 목록을 검색합니다. 카테고리: 프로그래밍, 데이터 과학, 인공지능",
            "parameters": {
                "type": "object",
                "properties": {
                    "category": {
                        "type": "string",
                        "description": "검색할 카테고리명"
                    }
                },
                "required": ["category"]
            }
        }
    }
]
```


---

## 3. Tool 호출 처리

LLM이 tool 호출을 요청하면, 해당 함수를 실행하고 결과를 반환하는 핸들러를 작성합니다.

### LLM 응답 구조 이해하기

LLM이 Tool 호출이 필요하다고 판단하면, 다음과 같은 구조의 응답을 반환합니다:

```python
# 전체 응답 구조
response.choices[0] = {
    "finish_reason": "tool_calls",  # ← Tool 호출이 필요함!
    "index": 0,
    "message": {
        "role": "assistant",
        "content": None,  # Tool 호출 시 content는 비어있음
        "tool_calls": [
            {
                "id": "call_abc123",           # 각 호출의 고유 ID
                "type": "function",
                "function": {
                    "name": "get_book_info",   # 호출할 함수 이름
                    "arguments": "{\"title\": \"클린 코드\"}"  # JSON 문자열로 된 인자
                }
            }
        ]
    }
}
```

#### finish_reason 값의 종류

| finish_reason | 의미 | 다음 행동 |
|---------------|------|----------|
| `"tool_calls"` | Tool 호출이 필요함 (미완성) | 함수를 실행하고 결과를 다시 전송 |
| `"stop"` | 정상적으로 응답 완료 | 최종 응답을 사용자에게 반환 |
| `"length"` | 최대 토큰 수 도달 | 응답이 잘렸으므로 처리 필요 |
| `"content_filter"` | 콘텐츠 필터에 차단됨 | 에러 처리 필요 |

**중요**: `finish_reason`은 `message` 객체가 아니라 `choice` 객체에 위치합니다!

```python
# ✅ 올바른 접근
response.choices[0].finish_reason

# ❌ 잘못된 접근
response.choices[0].message.finish_reason  # 존재하지 않음!
```

### Tool 응답 형식

함수 실행 결과를 LLM에 전달할 때는 반드시 다음 형식을 따라야 합니다:

```python
{
    "role": "tool",                    # 반드시 "tool"
    "content": "함수 실행 결과 문자열",   # 결과는 문자열이어야 함
    "tool_call_id": "call_abc123"      # 어떤 호출에 대한 응답인지 매칭
}
```


```python
# Tool 이름과 실제 함수를 매핑
available_tools = {
    "get_book_info": get_book_info,
    "check_stock": check_stock,
    "search_by_category": search_by_category
}

def handle_tool_calls(message):
    """LLM의 tool 호출 요청을 처리합니다."""
    responses = []
    
    for tool_call in message.tool_calls:
        function_name = tool_call.function.name
        arguments = json.loads(tool_call.function.arguments)
        
        # 해당 함수 실행
        if function_name in available_tools:
            result = available_tools[function_name](**arguments)
        else:
            result = f"Unknown function: {function_name}"
        
        # Tool 응답 형식으로 반환
        responses.append({
            "role": "tool",
            "content": result,
            "tool_call_id": tool_call.id
        })
    
    return responses
```


### handle_tool_calls 코드 분석

```python
def handle_tool_calls(message):
    responses = []
    
    for tool_call in message.tool_calls:      # ① 여러 Tool 호출을 순회
        function_name = tool_call.function.name        # ② 함수 이름 추출
        arguments = json.loads(tool_call.function.arguments)  # ③ JSON → dict 변환
        
        if function_name in available_tools:
            result = available_tools[function_name](**arguments)  # ④ 함수 실행
        else:
            result = f"Unknown function: {function_name}"
        
        responses.append({
            "role": "tool",
            "content": result,                 # ⑤ 결과는 문자열
            "tool_call_id": tool_call.id       # ⑥ 호출 ID 매칭 (필수!)
        })
    
    return responses
```

| 단계 | 설명 |
|------|------|
| ① | LLM이 한 번에 여러 Tool을 호출할 수 있으므로 반복문으로 처리 |
| ② | `tool_call.function.name`에서 호출할 함수 이름 추출 |
| ③ | `arguments`는 JSON 문자열이므로 `json.loads()`로 딕셔너리로 변환 |
| ④ | `**arguments`로 딕셔너리를 함수 인자로 언패킹하여 실행 |
| ⑤ | 함수 실행 결과는 반드시 **문자열**이어야 함 |
| ⑥ | `tool_call_id`는 어떤 호출에 대한 응답인지 LLM에게 알려주는 필수 값 |

#### [참고] `**arguments` 란?

Python의 **딕셔너리 언패킹(unpacking)** 문법입니다.

```python
arguments = {"title": "클린 코드"}

# ** 없이 전달하면
get_book_info({"title": "클린 코드"})  # ❌ 딕셔너리 객체 하나가 전달됨

# ** 로 전달하면
get_book_info(**arguments)             # ✅ 아래와 동일
get_book_info(title="클린 코드")       # ✅ 키=값 형태로 풀려서 전달됨
```

LLM이 반환하는 `arguments`는 딕셔너리이고, 함수는 `func(key=value)` 형태로 인자를 받으므로 `**`로 풀어서 전달합니다.

> **주의**: `tool_call_id`를 누락하면 LLM이 어떤 Tool 호출에 대한 결과인지 알 수 없어 오류가 발생합니다.

---

## 4. 채팅 함수 구현

Tool 호출을 포함한 전체 채팅 로직을 구현합니다.


```python
system_message = """당신은 온라인 서점 '북스토어'의 AI 도우미입니다.

[역할]
- 도서 정보 안내
- 재고 확인
- 카테고리별 도서 추천

[응대 지침]
- 친절하고 간결하게 응답하세요
- 도서 정보가 필요하면 제공된 도구를 활용하세요
- 찾는 책이 없으면 비슷한 책을 추천해주세요"""

def chat(message, history):
    # 대화 히스토리 구성
    history = [{"role":h["role"], "content":h["content"]} for h in history]
    messages = [{"role": "system", "content": system_message}] + history + [{"role": "user", "content": message}]

    
    # LLM 호출 (tools 전달)
    response = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        tools=tools
    )
    
    # Tool 호출이 필요한 경우 반복 처리
    while response.choices[0].finish_reason == "tool_calls":
        assistant_message = response.choices[0].message
        tool_responses = handle_tool_calls(assistant_message)
        
        # Tool 호출과 결과를 메시지에 추가
        messages.append(assistant_message)
        messages.extend(tool_responses)
        
        # 다시 LLM 호출
        response = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            tools=tools
        )
    
    return response.choices[0].message.content
```


### 왜 `while` 루프인가?

`while response.choices[0].finish_reason == "tool_calls":` 코드가 필요한 이유는 **LLM이 여러 번 Tool을 호출할 수 있기 때문**입니다.

#### finish_reason 값의 의미

| finish_reason | 의미 |
|---------------|------|
| `"stop"` | 최종 응답 완료, 더 이상 할 일 없음 |
| `"tool_calls"` | Tool 호출 필요, 아직 응답 미완성 |

#### 단일 요청에 여러 Tool 호출

```
사용자: "클린 코드 책 정보랑 재고도 알려줘"
```

```
1차 LLM 호출 → finish_reason = "tool_calls"
               tool_calls = [get_book_info("클린 코드"), check_stock("클린 코드")]
               
Tool 실행 후 2차 LLM 호출 → finish_reason = "stop"
                            content = "클린 코드는 로버트 마틴 저자이며..."
```

#### 연쇄적 Tool 호출 (Agentic 패턴)

```
사용자: "프로그래밍 책 중에서 재고 있는 거 알려줘"
```

```
1차: search_by_category("프로그래밍") 
     → 결과: "파이썬 코딩의 기술, 클린 코드"
     
2차: check_stock("파이썬 코딩의 기술"), check_stock("클린 코드")
     → 결과: 각 책의 재고 정보
     
3차: finish_reason = "stop" → 최종 응답 생성
```

LLM이 **카테고리 검색 결과를 본 후** 각 책의 재고를 확인하기로 스스로 결정합니다.

#### if vs while 비교

```python
# ❌ if: 한 번만 처리 - 연쇄 호출 불가
if response.choices[0].finish_reason == "tool_calls":
    ...

# ✅ while: 반복 처리 - 연쇄 호출 가능
while response.choices[0].finish_reason == "tool_calls":
    ...
```

이 패턴이 **Agentic AI**의 기초입니다. LLM이 스스로 판단하여 필요한 만큼 Tool을 호출하고, 최종 응답을 생성합니다.


```python
# Gradio 채팅 UI 실행
demo = gr.ChatInterface(
    fn=chat,
    title="북스토어 AI 도우미",
    description="도서 정보, 재고, 카테고리별 검색을 도와드립니다.",
    examples=[
        "클린 코드 책 정보 알려줘",
        "딥러닝 입문 재고 있어?",
        "프로그래밍 관련 책 추천해줘"
    ]
)

demo.launch()
```



<div class="nb-output">

```text
* Running on local URL:  http://127.0.0.1:7864
* To create a public link, set `share=True` in `launch()`.
<IPython.core.display.HTML object>
[Tool 호출] get_book_info('클린 코드')
```

</div>


---

## 5. 실전: SQLite 연동

실제 서비스에서는 딕셔너리가 아닌 데이터베이스에서 정보를 조회합니다. SQLite를 사용하여 동일한 기능을 구현해봅니다.


```python
import sqlite3

DB_PATH = "bookstore.db"

# 테이블 생성
with sqlite3.connect(DB_PATH) as conn:
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS books (
            title TEXT PRIMARY KEY,
            author TEXT,
            price INTEGER,
            stock INTEGER,
            category TEXT
        )
    ''')
    conn.commit()

# 초기 데이터 삽입
def init_db():
    books = [
        ("파이썬 코딩의 기술", "브렛 슬라킨", 32000, 15, "프로그래밍"),
        ("클린 코드", "로버트 마틴", 33000, 8, "프로그래밍"),
        ("데이터 과학을 위한 통계", "피터 브루스", 28000, 0, "데이터 과학"),
        ("딥러닝 입문", "사이토 고키", 24000, 23, "인공지능"),
    ]
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        for book in books:
            cursor.execute('''
                INSERT OR REPLACE INTO books (title, author, price, stock, category)
                VALUES (?, ?, ?, ?, ?)
            ''', book)
        conn.commit()

init_db()
print("DB 초기화 완료!")
```



<div class="nb-output">

```text
DB 초기화 완료!
```

</div>


### DB 조회 함수

딕셔너리 대신 SQLite에서 데이터를 조회하는 함수들입니다.


```python
def get_book_info_db(title: str) -> str:
    """DB에서 도서 정보를 조회합니다."""
    print(f"[DB Tool 호출] get_book_info_db('{title}')")
    
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        cursor.execute(
            'SELECT author, price, category FROM books WHERE title = ?', 
            (title,)
        )
        result = cursor.fetchone()
    
    if result:
        author, price, category = result
        return f"'{title}' - 저자: {author}, 가격: {price:,}원, 카테고리: {category}"
    return f"'{title}' 도서를 찾을 수 없습니다."

def check_stock_db(title: str) -> str:
    """DB에서 도서 재고를 확인합니다."""
    print(f"[DB Tool 호출] check_stock_db('{title}')")
    
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        cursor.execute('SELECT stock FROM books WHERE title = ?', (title,))
        result = cursor.fetchone()
    
    if result:
        stock = result[0]
        if stock > 0:
            return f"'{title}' 재고: {stock}권 (구매 가능)"
        return f"'{title}' 현재 품절입니다."
    return f"'{title}' 도서를 찾을 수 없습니다."

def search_by_category_db(category: str) -> str:
    """DB에서 카테고리별 도서를 검색합니다."""
    print(f"[DB Tool 호출] search_by_category_db('{category}')")
    
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        cursor.execute('SELECT title FROM books WHERE category = ?', (category,))
        results = cursor.fetchall()
    
    if results:
        titles = [row[0] for row in results]
        return f"{category} 카테고리 도서: {', '.join(titles)}"
    return f"{category} 카테고리에 해당하는 도서가 없습니다."

# 함수 테스트
print(get_book_info_db("클린 코드"))
print(check_stock_db("데이터 과학을 위한 통계"))
print(search_by_category_db("프로그래밍"))
```



<div class="nb-output">

```text
[DB Tool 호출] get_book_info_db('클린 코드')
'클린 코드' - 저자: 로버트 마틴, 가격: 33,000원, 카테고리: 프로그래밍
[DB Tool 호출] check_stock_db('데이터 과학을 위한 통계')
'데이터 과학을 위한 통계' 현재 품절입니다.
[DB Tool 호출] search_by_category_db('프로그래밍')
프로그래밍 카테고리 도서: 파이썬 코딩의 기술, 클린 코드
```

</div>


### DB 버전 채팅 함수

Tool 매핑과 핸들러를 DB 함수로 교체합니다.


```python
# DB 버전 Tool 매핑
available_tools_db = {
    "get_book_info": get_book_info_db,
    "check_stock": check_stock_db,
    "search_by_category": search_by_category_db
}

def handle_tool_calls_db(message):
    """DB 함수를 사용하는 Tool 호출 핸들러"""
    responses = []
    
    for tool_call in message.tool_calls:
        function_name = tool_call.function.name
        arguments = json.loads(tool_call.function.arguments)
        
        if function_name in available_tools_db:
            result = available_tools_db[function_name](**arguments)
        else:
            result = f"Unknown function: {function_name}"
        
        responses.append({
            "role": "tool",
            "content": result,
            "tool_call_id": tool_call.id
        })
    
    return responses

def chat_with_db(message, history):
    """SQLite DB를 사용하는 채팅 함수"""
    history = [{"role":h["role"], "content":h["content"]} for h in history]
    messages = [{"role": "system", "content": system_message}] + history + [{"role": "user", "content": message}]

    response = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        tools=tools  # 기존 스키마 재사용
    )
    
    while response.choices[0].finish_reason == "tool_calls":
        assistant_message = response.choices[0].message
        tool_responses = handle_tool_calls_db(assistant_message)  # DB 핸들러 사용
        
        messages.append(assistant_message)
        messages.extend(tool_responses)
        
        response = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            tools=tools
        )
    
    return response.choices[0].message.content
```



```python
# SQLite 연동 버전 실행
demo_db = gr.ChatInterface(
    fn=chat_with_db,
    title="북스토어 AI 도우미 (SQLite 연동)",
    description="SQLite DB에서 도서 정보를 조회합니다.",
    examples=[
        "클린 코드 책 정보 알려줘",
        "인공지능 관련 책 있어?",
        "데이터 과학을 위한 통계 재고 확인해줘"
    ]
)

demo_db.launch()
```



<div class="nb-output">

```text
* Running on local URL:  http://127.0.0.1:7865
* To create a public link, set `share=True` in `launch()`.
<IPython.core.display.HTML object>
[DB Tool 호출] get_book_info_db('클린 코드')
```

</div>


### 딕셔너리 vs SQLite 비교

| 구분 | 딕셔너리 | SQLite |
|------|----------|--------|
| 장점 | 간단, 빠름 | 영속성, 대용량 데이터, SQL 쿼리 |
| 단점 | 메모리 한계, 프로그램 종료 시 소멸 | 설정 필요 |
| 용도 | 프로토타입, 테스트 | 실제 서비스 |

**핵심 포인트**: Tool 함수 내부 구현만 변경하면 됩니다. LLM에게 전달하는 tools 스키마는 동일하게 유지됩니다.

---

## Tool 호출 흐름 정리

```
1. 사용자 메시지 + tools 스키마 → LLM 호출
    ↓
2. LLM이 finish_reason="tool_calls" 반환
    ↓
3. tool_calls에서 함수명, 인자 추출
    ↓
4. 해당 함수 실행 → 결과 획득
    ↓
5. tool 응답을 messages에 추가
    ↓
6. 다시 LLM 호출 → 최종 응답 생성
```

### 주의사항

- **보안**: Tool 함수에서 사용자 입력을 그대로 사용하지 말고 검증하세요
- **에러 처리**: 함수 실행 실패 시 적절한 에러 메시지를 반환하세요
- **무한 루프 방지**: Tool 호출 횟수에 제한을 두세요

---

## 요약

이번 노트북에서는 LLM의 Tool Use(Function Calling) 기능을 알아보았습니다.

### 핵심 포인트

1. **Tool 정의**: 함수와 JSON Schema 스키마 작성
2. **Tool 전달**: `tools` 파라미터로 LLM에 전달
3. **호출 감지**: `finish_reason == "tool_calls"` 확인
4. **결과 반환**: `role: "tool"` 형식으로 결과 전달
5. **반복 처리**: 여러 Tool 호출을 while 루프로 처리
