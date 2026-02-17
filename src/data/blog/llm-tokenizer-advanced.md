---
title: "LLM Tokenizer 추가 학습"
author: iwindfree
pubDatetime: 2025-01-07T09:00:00Z
slug: "llm-tokenizer-advanced"
category: "LLM Engineering"
series: "LLM Engineering"
seriesOrder: 2
tags: ["ai", "llm", "tokenization"]
description: "from huggingface_hub import login"
---


```python
from huggingface_hub import login
from transformers import AutoTokenizer
```



<div class="nb-output">

```text
/Users/windfree/.pyenv/versions/3.13.2/lib/python3.13/site-packages/tqdm/auto.py:21: TqdmWarning: IProgress not found. Please update jupyter and ipywidgets. See https://ipywidgets.readthedocs.io/en/stable/user_install.html
  from .autonotebook import tqdm as notebook_tqdm
PyTorch was not found. Models won't be available and only tokenizers, configuration and file/data utilities can be used.
```

</div>


## 소개

대규모 언어 모델(LLM)을 이해하는 데 있어 **토크나이저(Tokenizer)** 는 가장 기본적이면서도 중요한 구성 요소입니다. 토크나이저는 인간이 이해하는 텍스트를 모델이 처리할 수 있는 숫자로 변환하는 핵심 역할을 담당합니다.

### 토크나이저가 하는 일

1. **텍스트 → 토큰**: 입력 텍스트를 의미 있는 단위(토큰)로 분할
2. **토큰 → ID**: 각 토큰을 고유한 정수 ID로 변환
3. **ID → 토큰**: 모델 출력(ID)을 다시 토큰으로 변환
4. **토큰 → 텍스트**: 토큰을 다시 읽을 수 있는 텍스트로 복원

### 왜 토크나이저가 중요한가?

- **효율성**: 단어 단위보다 더 효율적인 하위 단어(subword) 단위 처리
- **일관성**: 학습과 추론 시 동일한 방식으로 텍스트 처리
- **어휘 관리**: 제한된 vocabulary로 무한한 텍스트 표현
- **다국어 지원**: 다양한 언어를 효과적으로 처리


```python
import os
```



```python
import tiktoken
```



```python
from huggingface_hub import login
from transformers import AutoTokenizer
from dotenv import load_dotenv
```



```python
# HuggingFace 로그인 (선택사항 - 일부 제한된 모델에만 필요)
# 공개 모델을 사용할 경우 이 단계는 건너뛰어도 됩니다
#hf_token = "add user tocken here"
hf_token = os.getenv("HF_TOKEN")

#load_dotenv(override=True)
from huggingface_hub import login
login(hf_token)
```


## 실습 준비

먼저 필요한 라이브러리를 import하고 토크나이저를 준비하겠습니다.


```python
# Llama 3.1 토크나이저 로드
tokenizer = AutoTokenizer.from_pretrained('meta-llama/Meta-Llama-3.1-8B', trust_remote_code=True)
print("토크나이저 로드 완료!")
```



<div class="nb-output">

```text
토크나이저 로드 완료!
```

</div>



```python
# 간단한 영어 문장 토큰화
text = "Hello, how are you today?"
#tokens = tokenizer.tokenize(text)

encoding = tiktoken.encoding_for_model("gpt-4")
tokens = encoding.encode(text)

print(f"원본 텍스트: {text}")
print(f"토큰 개수: {len(tokens)}")
print(f"토큰 리스트: {tokens}")
```



<div class="nb-output">

```text
원본 텍스트: Hello, how are you today?
토큰 개수: 7
토큰 리스트: [9906, 11, 1268, 527, 499, 3432, 30]
```

</div>



```python
# 복잡한 예제: 코드와 특수문자
code_text = "def hello_world():\n    print('Hello, World!')"
code_tokens = tokenizer.tokenize(code_text)

print(f"원본 코드:\n{code_text}\n")
print(f"토큰 개수: {len(code_tokens)}")
print(f"토큰 리스트: {code_tokens}")
```



<div class="nb-output">

```text
원본 코드:
def hello_world():
    print('Hello, World!')

토큰 개수: 12
토큰 리스트: ['def', 'Ġhello', '_world', '():Ċ', 'ĠĠĠ', 'Ġprint', "('", 'Hello', ',', 'ĠWorld', '!', "')"]
```

</div>


## 특수 토큰 (Special Tokens)

LLM은 텍스트 처리를 위해 여러 특수 토큰을 사용합니다:

- **BOS (Beginning of Sequence)**: 시퀀스의 시작을 표시
- **EOS (End of Sequence)**: 시퀀스의 끝을 표시  
- **PAD (Padding)**: 배치 처리 시 길이를 맞추기 위한 패딩
- **UNK (Unknown)**: 어휘에 없는 단어를 표시

이러한 특수 토큰들은 모델이 텍스트의 구조를 이해하는 데 중요한 역할을 합니다.


```python
# 특수 토큰 확인
print("=== 특수 토큰 정보 ===")
print(f"BOS 토큰: {tokenizer.bos_token} (ID: {tokenizer.bos_token_id})")
print(f"EOS 토큰: {tokenizer.eos_token} (ID: {tokenizer.eos_token_id})")
print(f"PAD 토큰: {tokenizer.pad_token} (ID: {tokenizer.pad_token_id})")
print(f"UNK 토큰: {tokenizer.unk_token} (ID: {tokenizer.unk_token_id})")
print(f"\n모든 특수 토큰: {tokenizer.special_tokens_map}")
```



<div class="nb-output">

```text
=== 특수 토큰 정보 ===
BOS 토큰: <|begin_of_text|> (ID: 128000)
EOS 토큰: <|end_of_text|> (ID: 128001)
PAD 토큰: None (ID: None)
UNK 토큰: None (ID: None)

모든 특수 토큰: {'bos_token': '<|begin_of_text|>', 'eos_token': '<|end_of_text|>'}
```

</div>



```python
# 특수 토큰이 포함된 텍스트 처리
text_with_special = "Hello, World!"
# add_special_tokens=True로 BOS/EOS 토큰 자동 추가
encoded_with_special = tokenizer.encode(text_with_special, add_special_tokens=True)
encoded_without_special = tokenizer.encode(text_with_special, add_special_tokens=False)

print(f"원본 텍스트: {text_with_special}")
print(f"\n특수 토큰 포함 (add_special_tokens=True):")
print(f"  토큰 ID: {encoded_with_special}")
print(f"  토큰 개수: {len(encoded_with_special)}")
print(f"\n특수 토큰 제외 (add_special_tokens=False):")
print(f"  토큰 ID: {encoded_without_special}")
print(f"  토큰 개수: {len(encoded_without_special)}")
```



<div class="nb-output">

```text
원본 텍스트: Hello, World!

특수 토큰 포함 (add_special_tokens=True):
  토큰 ID: [128000, 9906, 11, 4435, 0]
  토큰 개수: 5

특수 토큰 제외 (add_special_tokens=False):
  토큰 ID: [9906, 11, 4435, 0]
  토큰 개수: 4
```

</div>


## 인코딩과 디코딩

토크나이저의 핵심 기능은 텍스트를 ID로 변환(인코딩)하고, ID를 다시 텍스트로 복원(디코딩)하는 것입니다.


```python
# __call__ 메서드를 사용한 고급 인코딩
# 이 방법이 더 많은 기능을 제공합니다
text = "Tokenization is the first step in LLM processing."

tokens = tokenizer.encode(text)
tokens
```



<div class="nb-output">

```text
[128000, 3404, 2065, 374, 279, 1176, 3094, 304, 445, 11237, 8863, 13]
```

</div>



```python
character_count = len(text)
word_count = len(text.split(' '))
token_count = len(tokens)
print(f"There are {character_count} characters, {word_count} words and {token_count} tokens")
```



<div class="nb-output">

```text
There are 49 characters, 8 words and 12 tokens
```

</div>



```python
tokenizer.decode(tokens)
```



<div class="nb-output">

```text
'<|begin_of_text|>Tokenization is the first step in LLM processing.'
```

</div>



```python
tokenizer.batch_decode(tokens)
```



<div class="nb-output">

```text
['<|begin_of_text|>Tokenization is the first step in LLM processing.']
```

</div>


## Vocabulary와 고급 개념

토크나이저의 vocabulary는 모델이 이해할 수 있는 모든 토큰의 집합입니다.


```python
# 특정 토큰 ID와 텍스트 간 변환
token_ids = [128000, 9906, 11, 1917]  # 임의의 토큰 ID들

print("=== 토큰 ID ↔ 텍스트 변환 ===")
for token_id in token_ids:
    # ID → 토큰
    token = tokenizer.convert_ids_to_tokens(token_id)
    # ID → 텍스트 (디코딩)
    text = tokenizer.decode([token_id])
    print(f"ID {token_id} → 토큰: '{token}' → 텍스트: '{text}'")
```



<div class="nb-output">

```text
=== 토큰 ID ↔ 텍스트 변환 ===
ID 128000 → 토큰: '<|begin_of_text|>' → 텍스트: '<|begin_of_text|>'
ID 9906 → 토큰: 'Hello' → 텍스트: 'Hello'
ID 11 → 토큰: ',' → 텍스트: ','
ID 1917 → 토큰: 'Ġworld' → 텍스트: ' world'
```

</div>



```python
# 서브워드 토큰화 원리 이해
words = ["tokenization", "antidisestablishmentarianism", "AI", "🤖", "café"]

print("=== 서브워드 토큰화 예제 ===\n")
for word in words:
    tokens = tokenizer.tokenize(word)
    print(f"단어: '{word}'")
    print(f"  토큰 개수: {len(tokens)}")
    print(f"  토큰: {tokens}")
    print()
```



<div class="nb-output">

```text
=== 서브워드 토큰화 예제 ===

단어: 'tokenization'
  토큰 개수: 2
  토큰: ['token', 'ization']

단어: 'antidisestablishmentarianism'
  토큰 개수: 6
  토큰: ['ant', 'idis', 'establish', 'ment', 'arian', 'ism']

단어: 'AI'
  토큰 개수: 1
  토큰: ['AI']

단어: '🤖'
  토큰 개수: 3
  토큰: ['ðŁ', '¤', 'ĸ']

단어: 'café'
  토큰 개수: 2
  토큰: ['ca', 'fÃ©']
```

</div>



```python
# tokenizer.vocab
tokenizer.get_added_vocab()
```



<div class="nb-output">

```text
{'<|begin_of_text|>': 128000,
 '<|end_of_text|>': 128001,
 '<|reserved_special_token_0|>': 128002,
 '<|reserved_special_token_1|>': 128003,
 '<|finetune_right_pad_id|>': 128004,
 '<|reserved_special_token_2|>': 128005,
 '<|start_header_id|>': 128006,
 '<|end_header_id|>': 128007,
 '<|eom_id|>': 128008,
 '<|eot_id|>': 128009,
 '<|python_tag|>': 128010,
 '<|reserved_special_token_3|>': 128011,
 '<|reserved_special_token_4|>': 128012,
 '<|reserved_special_token_5|>': 128013,
 '<|reserved_special_token_6|>': 128014,
 '<|reserved_special_token_7|>': 128015,
 '<|reserved_special_token_8|>': 128016,
 '<|reserved_special_token_9|>': 128017,
 '<|reserved_special_token_10|>': 128018,
 '<|reserved_special_token_11|>': 128019,
 '<|reserved_special_token_12|>': 128020,
 '<|reserved_special_token_13|>': 128021,
 '<|reserved_special_token_14|>': 128022,
 '<|reserved_special_token_15|>': 128023,
 '<|reserved_special_token_16|>': 128024,
 '<|reserved_special_token_17|>': 128025,
 '<|reserved_special_token_18|>': 128026,
 '<|reserved_special_token_19|>': 128027,
 '<|reserved_special_token_20|>': 128028,
 '<|reserved_special_token_21|>': 128029,
 '<|reserved_special_token_22|>': 128030,
 '<|reserved_special_token_23|>': 128031,
 '<|reserved_special_token_24|>': 128032,
 '<|reserved_special_token_25|>': 128033,
 '<|reserved_special_token_26|>': 128034,
 '<|reserved_special_token_27|>': 128035,
 '<|reserved_special_token_28|>': 128036,
 '<|reserved_special_token_29|>': 128037,
 '<|reserved_special_token_30|>': 128038,
 '<|reserved_special_token_31|>': 128039,
 '<|reserved_special_token_32|>': 128040,
 '<|reserved_special_token_33|>': 128041,
 '<|reserved_special_token_34|>': 128042,
 '<|reserved_special_token_35|>': 128043,
 '<|reserved_special_token_36|>': 128044,
 '<|reserved_special_token_37|>': 128045,
 '<|reserved_special_token_38|>': 128046,
 '<|reserved_special_token_39|>': 128047,
 '<|reserved_special_token_40|>': 128048,
 '<|reserved_special_token_41|>': 128049,
... (출력 206줄 생략)
```

</div>



```python
# Vocabulary 크기 확인
vocab_size = tokenizer.vocab_size

print(f"=== Vocabulary 정보 ===")
print(f"Vocabulary 크기: {vocab_size:,}")
print(f"모델 최대 길이: {tokenizer.model_max_length:,} 토큰")
print(f"\n이 토크나이저는 {vocab_size:,}개의 서로 다른 토큰을 구분할 수 있습니다.")
```



<div class="nb-output">

```text
=== Vocabulary 정보 ===
Vocabulary 크기: 128,000
모델 최대 길이: 131,072 토큰

이 토크나이저는 128,000개의 서로 다른 토큰을 구분할 수 있습니다.
```

</div>


## 결론

이 노트북에서 다룬 내용:

1. ✅ **토크나이저의 기본 개념**: 텍스트를 숫자로 변환하는 핵심 역할
2. ✅ **토큰화 과정**: tokenize, encode, decode 메서드 사용법
3. ✅ **특수 토큰**: BOS, EOS, PAD, UNK의 역할과 사용
4. ✅ **배치 처리**: padding과 attention mask를 활용한 효율적인 처리
5. ✅ **고급 개념**: vocabulary, truncation, 서브워드 토큰화
6. ✅ **모델 비교**: 서로 다른 토크나이저의 특성 비교
7. ✅ **실전 활용**: API 비용 계산, 컨텍스트 윈도우 관리

### 핵심 포인트

- 토크나이저는 LLM의 "언어"를 정의합니다
- 같은 텍스트도 모델에 따라 다르게 토큰화됩니다
- 토큰 수는 비용과 성능에 직접적인 영향을 미칩니다
- 실무에서는 항상 토큰 수를 모니터링하고 관리해야 합니다


```python
# 컨텍스트 윈도우 관리
def check_context_limit(text, max_tokens=4096):
    """
    텍스트가 컨텍스트 제한을 초과하는지 확인
    
    Args:
        text: 확인할 텍스트
        max_tokens: 최대 토큰 수
    
    Returns:
        초과 여부와 정보
    """
    tokens = tokenizer.encode(text)
    num_tokens = len(tokens)
    is_over = num_tokens > max_tokens
    
    return {
        'num_tokens': num_tokens,
        'max_tokens': max_tokens,
        'is_over_limit': is_over,
        'remaining': max_tokens - num_tokens,
        'percentage': (num_tokens / max_tokens) * 100
    }

# 테스트
long_text = "This is a test sentence. " * 200
result = check_context_limit(long_text, max_tokens=128)

print("=== 컨텍스트 윈도우 체크 ===")
print(f"토큰 수: {result['num_tokens']}")
print(f"최대 허용: {result['max_tokens']}")
print(f"제한 초과: {'예' if result['is_over_limit'] else '아니오'}")
print(f"사용률: {result['percentage']:.1f}%")

if result['is_over_limit']:
    print(f"⚠️  {abs(result['remaining'])} 토큰 초과! 텍스트를 줄여야 합니다.")
else:
    print(f"✓ {result['remaining']} 토큰 여유 있음")
```



```python
# Truncation 예제: 긴 텍스트 처리
long_text = "Large Language Models " * 100  # 매우 긴 반복 텍스트

# truncation 없이 (경고 발생 가능)
encoded_no_trunc = tokenizer(long_text, truncation=False)
print(f"=== Truncation 테스트 ===")
print(f"원본 텍스트 길이: {len(long_text)} 문자")
print(f"Truncation 없음: {len(encoded_no_trunc['input_ids'])} 토큰")

# truncation 사용 (최대 길이로 자름)
encoded_with_trunc = tokenizer(long_text, truncation=True, max_length=50)
print(f"Truncation 사용 (max_length=50): {len(encoded_with_trunc['input_ids'])} 토큰")
print(f"\n모델의 최대 컨텍스트 길이를 초과하는 텍스트는 잘라내야 합니다.")
```



```python
# Vocabulary 크기 확인
vocab_size = tokenizer.vocab_size

print(f"=== Vocabulary 정보 ===")
print(f"Vocabulary 크기: {vocab_size:,}")
print(f"모델 최대 길이: {tokenizer.model_max_length:,} 토큰")
print(f"\n이 토크나이저는 {vocab_size:,}개의 서로 다른 토큰을 구분할 수 있습니다.")
```



<div class="nb-output">

```text
=== Vocabulary 정보 ===
Vocabulary 크기: 128,000
모델 최대 길이: 131,072 토큰

이 토크나이저는 128,000개의 서로 다른 토큰을 구분할 수 있습니다.
```

</div>



```python
# 기본 인코딩/디코딩
text = "Large Language Models are transforming AI!"

# 인코딩: 텍스트 → 토큰 ID
encoded = tokenizer.encode(text)
print(f"원본 텍스트: {text}")
print(f"인코딩 결과 (토큰 ID): {encoded}")

# 디코딩: 토큰 ID → 텍스트
decoded = tokenizer.decode(encoded)
print(f"디코딩 결과: {decoded}")
print(f"\n원본과 동일?: {text == decoded.strip()}")
```



<div class="nb-output">

```text
원본 텍스트: Large Language Models are transforming AI!
인코딩 결과 (토큰 ID): [128000, 35353, 11688, 27972, 527, 46890, 15592, 0]
디코딩 결과: <|begin_of_text|>Large Language Models are transforming AI!

원본과 동일?: False
```

</div>

