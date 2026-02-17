---
title: "Google Colab 사용해보기: 무료로 GPU 환경에서 AI 모델 실행하기"
author: iwindfree
pubDatetime: 2025-01-29T09:00:00Z
slug: "llm-using-google-colab"
category: "LLM Engineering"
series: "LLM Engineering"
seriesOrder: 12
tags: ["ai", "llm", "colab"]
description: "!Open In Colabhttps://colab.research.google.com/assets/colab-badge.svghttps://colab.research.google.com/github/your-repo/google_colab_gpu_guide.ipynb"
---

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/your-repo/google_colab_gpu_guide.ipynb)

## 개요

**Google Colab**은 Google이 제공하는 무료 클라우드 Jupyter Notebook 환경입니다. 별도의 설치 없이 브라우저에서 Python 코드를 실행할 수 있으며, **무료 GPU**를 제공합니다!

## 학습 목표

- ✅ Google Colab GPU 런타임 설정하기
- ✅ Colab Secrets으로 API 키 안전하게 관리하기
- ✅ Hugging Face 모델을 GPU로 실행하기
- ✅ OpenAI API를 Colab에서 사용하기
- ✅ CPU vs GPU 성능 비교하기

## Google Colab이란?

| 특징 | 설명 |
|------|------|
| **무료 GPU** | NVIDIA Tesla T4 GPU 무료 제공 |
| **설치 불필요** | 브라우저만 있으면 즉시 시작 |
| **Google Drive 연동** | 파일 저장 및 공유 간편 |
| **협업** | 링크 공유로 실시간 협업 가능 |
| **사전 설치된 라이브러리** | TensorFlow, PyTorch 등 이미 설치됨 |

### 무료 vs Colab Pro 비교

| 구분 | 무료 | Colab Pro | Colab Pro+ |
|------|------|-----------|------------|
| **가격** | $0 | $9.99/월 | $49.99/월 |
| **GPU** | T4 (제한적) | T4, V100, A100 | V100, A100 (우선 할당) |
| **세션 시간** | 12시간 | 24시간 | 24시간 |
| **유휴 시간** | 90분 | 더 긴 유휴 허용 | 더 긴 유휴 허용 |
| **백그라운드 실행** | ❌ | ✅ | ✅ |

> 💡 **학습 및 실험 목적**이라면 무료 버전으로도 충분합니다!

---

## 1. GPU 런타임 설정하기

Colab은 기본적으로 CPU 런타임으로 시작됩니다. GPU를 사용하려면 런타임 유형을 변경해야 합니다.

### 📌 GPU 활성화 단계

1. 상단 메뉴: **런타임(Runtime)** → **런타임 유형 변경(Change runtime type)**
2. **하드웨어 가속기(Hardware accelerator)**: **GPU** 선택
3. **GPU 유형**: T4 (무료 버전은 자동 선택)
4. **저장(Save)** 클릭

런타임이 재시작되며 GPU가 할당됩니다.

### GPU 할당 확인


```python
# GPU 정보 확인 (NVIDIA System Management Interface)
!nvidia-smi
```


**출력 예시:**
```
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 525.85.12    Driver Version: 525.85.12    CUDA Version: 12.0     |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|===============================+======================+======================|
|   0  Tesla T4            Off  | 00000000:00:04.0 Off |                    0 |
| N/A   36C    P8     9W /  70W |      0MiB / 15360MiB |      0%      Default |
+-------------------------------+----------------------+----------------------+
```

- **GPU 이름**: Tesla T4
- **VRAM**: 약 15GB
- **현재 사용량**: 0MiB (초기 상태)


```python
# PyTorch로 GPU 사용 가능 여부 확인
import torch

if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"✅ GPU 사용 가능!")
    print(f"📦 GPU 이름: {torch.cuda.get_device_name(0)}")
    print(f"💾 VRAM 총량: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
else:
    device = torch.device("cpu")
    print("❌ GPU를 사용할 수 없습니다. 런타임 유형을 확인하세요.")
```


---

## 2. Colab Secrets으로 API 키 안전하게 관리하기 🔑

API 키를 코드에 직접 입력하면 **보안 위험**이 있습니다. Google Colab의 **Secrets** 기능을 사용하면 안전하게 키를 관리할 수 있습니다.

### 📌 Colab Secrets 사용 방법

#### 1단계: Secrets 추가하기

1. **좌측 사이드바**의 **열쇠 아이콘(🔑)** 클릭
2. **+ Add new secret** 버튼 클릭
3. 다음 정보 입력:

**Hugging Face Token:**
- **Name**: `HF_TOKEN`
- **Value**: `hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx` (본인의 토큰)

**OpenAI API Key:**
- **Name**: `OPENAI_API_KEY`
- **Value**: `sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx` (본인의 키)

4. **저장 후 중요!** → 각 Secret 옆의 **"노트북 액세스" 토글을 켜기** ⚠️

#### 2단계: Python에서 Secrets 사용하기

```python
from google.colab import userdata

# Secrets에서 값 가져오기
hf_token = userdata.get('HF_TOKEN')
openai_key = userdata.get('OPENAI_API_KEY')
```

### 왜 Secrets을 사용해야 하나요?

| 방법 | 장점 | 단점 |
|------|------|------|
| **코드에 직접 입력** | 간편함 | ❌ 노트북 공유 시 키 노출 |
| **환경 변수 (.env)** | 로컬에서 안전 | ❌ Colab에서 파일 업로드 필요 |
| **Colab Secrets** | ✅ 안전 + 간편 | Colab 전용 |

> 💡 **Secrets는 런타임이 재시작되어도 유지**되며, 노트북을 공유해도 키는 노출되지 않습니다!


```python
# Secrets에서 API 키 로드하기
from google.colab import userdata

# Hugging Face Token 가져오기
try:
    HF_TOKEN = userdata.get('HF_TOKEN')
    print("✅ HF_TOKEN 로드 성공!")
    print(f"   토큰 앞 10자: {HF_TOKEN[:10]}...")
except Exception as e:
    print(f"❌ HF_TOKEN을 찾을 수 없습니다. Secrets에 추가했는지 확인하세요.")
    print(f"   에러: {e}")

# OpenAI API Key 가져오기
try:
    OPENAI_API_KEY = userdata.get('OPENAI_API_KEY')
    print("✅ OPENAI_API_KEY 로드 성공!")
    print(f"   키 앞 10자: {OPENAI_API_KEY[:10]}...")
except Exception as e:
    print(f"❌ OPENAI_API_KEY를 찾을 수 없습니다.")
    print(f"   에러: {e}")
```


---

## 3. 필수 라이브러리 설치

Colab에는 많은 라이브러리가 사전 설치되어 있지만, 최신 버전이 필요하거나 추가 패키지가 필요할 수 있습니다.


```python
# 필요한 라이브러리 설치 (이미 설치된 경우 업그레이드)
!pip install -q transformers accelerate huggingface_hub openai pillow
```



```python
# 설치된 버전 확인
import transformers
import torch
import openai

print(f"PyTorch 버전: {torch.__version__}")
print(f"CUDA 사용 가능: {torch.cuda.is_available()}")
print(f"Transformers 버전: {transformers.__version__}")
print(f"OpenAI 버전: {openai.__version__}")
```


---

## 4. 실습 예제 1: Hugging Face 모델 GPU로 실행하기

Hugging Face의 텍스트 생성 모델을 GPU에서 실행해봅시다.

### GPT-2로 텍스트 생성


```python
from transformers import pipeline
from google.colab import userdata
from huggingface_hub import login

# Hugging Face 로그인 (선택사항, 공개 모델은 불필요)
try:
    HF_TOKEN = userdata.get('HF_TOKEN')
    login(token=HF_TOKEN)
    print("✅ Hugging Face 로그인 성공!")
except:
    print("⚠️ HF_TOKEN이 없습니다. 공개 모델만 사용 가능합니다.")

# GPU를 사용하는 텍스트 생성 파이프라인 생성
# device=0: GPU 사용 (CPU는 device=-1)
generator = pipeline(
    "text-generation",
    model="gpt2",
    device=0  # GPU 사용
)

print("✅ 모델이 GPU에 로드되었습니다!")
print(f"   사용 중인 디바이스: {generator.device}")
```



```python
# 텍스트 생성 실행
prompt = "Artificial intelligence in healthcare will"

print(f"프롬프트: {prompt}\n")
print("생성 중...\n")

results = generator(
    prompt,
    max_length=100,
    num_return_sequences=3,
    temperature=0.9,
    do_sample=True
)

for i, result in enumerate(results, 1):
    print(f"=== 생성 결과 {i} ===")
    print(result['generated_text'])
    print()
```


### CPU vs GPU 성능 비교


```python
import time

# CPU 파이프라인
generator_cpu = pipeline("text-generation", model="gpt2", device=-1)

# GPU 파이프라인
generator_gpu = pipeline("text-generation", model="gpt2", device=0)

prompt = "The future of technology is"

# CPU 성능 측정
start_cpu = time.time()
_ = generator_cpu(prompt, max_length=50, num_return_sequences=1)
cpu_time = time.time() - start_cpu

# GPU 성능 측정
start_gpu = time.time()
_ = generator_gpu(prompt, max_length=50, num_return_sequences=1)
gpu_time = time.time() - start_gpu

print("⏱️ 성능 비교 (텍스트 생성)")
print("=" * 50)
print(f"CPU 소요 시간: {cpu_time:.3f}초")
print(f"GPU 소요 시간: {gpu_time:.3f}초")
print(f"속도 향상: {cpu_time/gpu_time:.2f}배 빠름")
print("=" * 50)
```


---

## 5. 실습 예제 2: OpenAI API 사용하기

Colab에서 OpenAI API를 사용해봅시다. Secrets에 저장된 API 키를 사용합니다.


```python
from openai import OpenAI
from google.colab import userdata

# Secrets에서 OpenAI API 키 가져오기
OPENAI_API_KEY = userdata.get('OPENAI_API_KEY')

# OpenAI 클라이언트 생성
client = OpenAI(api_key=OPENAI_API_KEY)

print("✅ OpenAI 클라이언트 초기화 완료!")
```



```python
# GPT-4o-mini로 대화하기
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {"role": "system", "content": "You are a helpful AI assistant."},
        {"role": "user", "content": "Explain Google Colab in one sentence."}
    ],
    temperature=0.7,
    max_tokens=100
)

print("💬 AI 응답:")
print(response.choices[0].message.content)
```


### 스트리밍 응답 받기


```python
# 스트리밍으로 실시간 응답 받기
print("💬 AI 응답 (스트리밍):")

stream = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {"role": "user", "content": "Write a short poem about machine learning."}
    ],
    stream=True
)

for chunk in stream:
    if chunk.choices[0].delta.content is not None:
        print(chunk.choices[0].delta.content, end="")

print()  # 줄바꿈
```


---

## 6. 실습 예제 3: 이미지 분류 (Vision Transformer)

GPU를 활용해 이미지 분류를 수행해봅시다.


```python
from transformers import pipeline
from PIL import Image
import requests
from io import BytesIO

# 이미지 분류 파이프라인 (GPU 사용)
image_classifier = pipeline("image-classification", device=0)

# 샘플 이미지 다운로드
url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg"
response = requests.get(url)
image = Image.open(BytesIO(response.content))

# 이미지 표시
display(image)

# 이미지 분류 실행
predictions = image_classifier(image)

print("\n🖼️ 이미지 분류 결과:")
for pred in predictions[:5]:
    print(f"  - {pred['label']}: {pred['score']:.4f}")
```


---

## 7. Google Drive 연동 (선택사항)

세션이 종료되면 Colab의 데이터는 사라집니다. Google Drive를 마운트하면 파일을 영구 저장할 수 있습니다.


```python
from google.colab import drive

# Google Drive 마운트
drive.mount('/content/drive')

print("✅ Google Drive가 /content/drive에 마운트되었습니다!")
print("   파일은 /content/drive/MyDrive/ 경로에 저장하세요.")
```



```python
# Drive에 파일 저장 예제
import os

# 작업 디렉토리 생성
save_dir = "/content/drive/MyDrive/colab_projects"
os.makedirs(save_dir, exist_ok=True)

# 파일 저장
with open(f"{save_dir}/test.txt", "w") as f:
    f.write("Hello from Colab!")

print(f"✅ 파일이 {save_dir}/test.txt에 저장되었습니다!")
```


---

## 8. GPU 메모리 모니터링

대형 모델을 실행할 때는 GPU 메모리를 모니터링해야 합니다.


```python
import torch

def print_gpu_memory():
    """현재 GPU 메모리 사용량 출력"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated(0) / 1e9
        reserved = torch.cuda.memory_reserved(0) / 1e9
        total = torch.cuda.get_device_properties(0).total_memory / 1e9
        
        print("📊 GPU 메모리 사용량")
        print(f"  할당됨: {allocated:.2f} GB")
        print(f"  예약됨: {reserved:.2f} GB")
        print(f"  전체: {total:.2f} GB")
        print(f"  사용률: {(reserved/total)*100:.1f}%")
    else:
        print("GPU를 사용할 수 없습니다.")

print_gpu_memory()
```



```python
# GPU 메모리 정리
import gc
import torch

# 변수 삭제
# del model, tokenizer  # 사용 중인 모델 삭제

# 가비지 컬렉션
gc.collect()

# PyTorch 캐시 비우기
torch.cuda.empty_cache()

print("✅ GPU 메모리가 정리되었습니다!")
print_gpu_memory()
```


---

## 9. 주의사항 및 팁 💡

### ⚠️ Colab 사용 제한사항

| 제한 | 내용 |
|------|------|
| **세션 시간** | 최대 12시간 (Pro: 24시간) |
| **유휴 시간** | 90분 동안 활동 없으면 자동 종료 |
| **GPU 할당** | 사용량이 많으면 GPU를 할당받지 못할 수 있음 |
| **동시 세션** | 무료는 1개, Pro는 여러 개 가능 |
| **파일 보존** | 런타임 종료 시 `/content` 데이터 삭제 |

### 💡 유용한 팁

#### 1. 세션 유지하기

```javascript
// 브라우저 콘솔에서 실행 (F12)
function KeepAlive() {
  console.log("Keeping session alive...");
  document.querySelector("colab-connect-button").click();
}
setInterval(KeepAlive, 60000); // 1분마다 실행
```

⚠️ **주의**: Google 정책 위반 가능성이 있으므로 신중하게 사용하세요.

#### 2. GPU 메모리 부족 에러 해결

```python
# 작은 배치 크기 사용
batch_size = 1  # 기본값보다 줄이기

# Mixed Precision 사용 (FP16)
# torch.cuda.amp 활용

# Gradient Checkpointing
model.gradient_checkpointing_enable()
```

#### 3. 모델 다운로드 속도 향상

```python
# 미러 사이트 사용
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
```

#### 4. 런타임 재시작 후 빠른 복구

```python
# 노트북 상단에 이 셀을 배치
# 런타임 재시작 시 이 셀만 실행하면 환경 복구

!pip install -q transformers accelerate huggingface_hub openai
from google.colab import userdata
HF_TOKEN = userdata.get('HF_TOKEN')
OPENAI_API_KEY = userdata.get('OPENAI_API_KEY')
```

#### 5. 큰 모델 사용 시 권장사항

```python
# 8bit 또는 4bit 양자화 사용
from transformers import AutoModelForCausalLM, BitsAndBytesConfig

quantization_config = BitsAndBytesConfig(
    load_in_8bit=True,
    # 또는 load_in_4bit=True
)

model = AutoModelForCausalLM.from_pretrained(
    "model_name",
    quantization_config=quantization_config,
    device_map="auto"
)
```

---

## 10. 결론 및 요약

### 🎓 배운 내용

✅ Google Colab GPU 런타임 설정하기  
✅ Colab Secrets으로 API 키 안전하게 관리하기  
✅ Hugging Face 모델을 GPU로 실행하기  
✅ OpenAI API를 Colab에서 사용하기  
✅ CPU vs GPU 성능 비교하기  
✅ GPU 메모리 모니터링 및 관리하기  

### 🔑 핵심 포인트

1. **Colab Secrets 사용**
   - 좌측 열쇠 아이콘(🔑) 클릭
   - API 키를 안전하게 저장
   - `userdata.get()으로 접근`

2. **GPU 활성화**
   - 런타임 → 런타임 유형 변경 → GPU 선택
   - `device=0` 또는 `device="cuda"`로 GPU 사용

3. **세션 관리**
   - 12시간 제한 (무료)
   - Google Drive 마운트로 데이터 보존
   - 중요한 파일은 Drive에 저장

### 📚 추가 학습 리소스

- [Google Colab 공식 문서](https://colab.research.google.com/notebooks/)
- [Hugging Face 모델 허브](https://huggingface.co/models)
- [PyTorch GPU 튜토리얼](https://pytorch.org/tutorials/beginner/blitz/tensor_tutorial.html#cuda-tensors)
- [Colab Pro 구독](https://colab.research.google.com/signup)



---

**🎉 수고하셨습니다!**

이제 Google Colab의 무료 GPU로 다양한 AI 모델을 실행할 수 있습니다. 실험을 즐기세요!
