---
title: "LLM을 활용한 회의록 자동 요약 시스템"
author: iwindfree
pubDatetime: 2025-01-21T09:00:00Z
slug: "llm-meeting-minutes-summary"
category: "LLM Engineering"
series: "LLM Engineering"
seriesOrder: 8
tags: ["ai", "llm", "prompting"]
description: "회의록 작성은 많은 조직에서 필수적이지만 시간이 많이 소요되는 작업입니다. 회의 내용을 정확하게 기록하고, 핵심 내용을 요약하며, 액션 아이템을 정리하는 데 회의 시간만큼의 시간이 걸릴 수 있습니다."
---

## 소개

회의록 작성은 많은 조직에서 필수적이지만 시간이 많이 소요되는 작업입니다. 회의 내용을 정확하게 기록하고, 핵심 내용을 요약하며, 액션 아이템을 정리하는 데 회의 시간만큼의 시간이 걸릴 수 있습니다.

### 문제점

- ⏰ **시간 소모**: 1시간 회의 = 30분~1시간의 회의록 작성
- 📝 **일관성 부족**: 작성자마다 다른 스타일과 포맷
- 🎯 **핵심 누락**: 중요한 결정 사항이나 액션 아이템 누락 가능
- 🔄 **반복 작업**: 매번 유사한 구조의 문서 작성

### AI 솔루션

이 노트북에서는 LLM(Large Language Model)을 활용하여 음성 회의를 자동으로 텍스트로 변환하고, 구조화된 회의록을 생성하는 전체 파이프라인을 구축합니다.

### 다룰 내용

1. 🎤 **Speech-to-Text**: OpenAI Whisper를 사용한 음성 → 텍스트 변환
2. 🤖 **LLM 활용**: 다양한 LLM을 사용한 회의록 요약
3. 📊 **구조화**: 요약, 논의 사항, 핵심 포인트, 액션 아이템 추출
4. ⚖️ **모델 비교**: Ollama(로컬) vs OpenAI(클라우드)
5. 💡 **실전 활용**: 프롬프트 최적화 및 자동화 팁


```python
import os
from dotenv import load_dotenv
from IPython.display import Markdown, display
from openai import OpenAI
```


## 필요한 라이브러리 설치 및 Import

먼저 필요한 라이브러리를 import합니다:
- `openai`: OpenAI API 사용 (Whisper, GPT)
- `dotenv`: 환경 변수 관리
- `IPython.display`: Jupyter에서 Markdown 렌더링

## 1단계: Speech to Text

### OpenAI Whisper 모델

**Whisper**는 OpenAI가 개발한 범용 음성 인식 모델입니다:

#### 주요 특징
- 🌍 **다국어 지원**: 99개 언어 지원
- 🎯 **높은 정확도**: 다양한 억양과 배경 소음 처리
- ⚡ **빠른 처리**: API를 통한 실시간 변환
- 💰 **합리적인 가격**: 1분당 $0.006

#### 지원 형식
- 파일 형식: mp3, mp4, mpeg, mpga, m4a, wav, webm
- 최대 크기: 25MB
- 긴 오디오는 청킹(chunking) 필요

이제 실제 음성 파일을 텍스트로 변환해보겠습니다.


```python
# OpenAI Whisper를 사용한 음성 → 텍스트 변환
AUDIO_MODEL = "gpt-4o-mini-transcribe"  # Whisper 모델
audio_file_path = "./denver_extract.mp3"  # 변환할 오디오 파일
audio_file = open(audio_file_path, "rb")

# 환경 변수에서 API 키 로드
load_dotenv(override=True) 
openai_api_key = os.getenv("OPENAI_API_KEY")
openai = OpenAI(api_key=openai_api_key)

# 음성 → 텍스트 변환
transcription = openai.audio.transcriptions.create(
    model=AUDIO_MODEL, 
    file=audio_file, 
    response_format="text"  # text, json, srt, verbose_json, vtt 중 선택
)

print("변환 완료! 결과:")
display(Markdown(transcription))
```



<div class="nb-output">

```text
변환 완료! 결과:
<IPython.core.display.Markdown object>
```

</div>


## 2단계: 프롬프트 엔지니어링으로 회의록 생성

이제 음성에서 텍스트로 전환된 내용을 기반으로 **구조화된 회의록**을 생성합니다.

### 프롬프트 설계 원칙

효과적인 회의록 생성을 위한 핵심 요소:

#### 1. 명확한 역할 정의 (System Message)
- LLM의 역할과 목표를 명확히 정의
- 출력 형식 지정 (마크다운, JSON 등)
- 일관된 품질 유지

#### 2. 구체적인 요구사항 (User Prompt)
- 원하는 정보 명시 (참석자, 일시, 장소)
- 섹션 구조 지정 (요약, 토론, 액션 아이템)
- 예시 제공으로 출력 품질 향상

#### 3. 컨텍스트 제공
- 회의 유형 (일반, 기술, 경영진 등)
- 조직 정보 (필요 시)
- 특별 요구사항

### 프롬프트 최적화 팁

**System Message 작성 팁:**
- 간결하고 명확하게
- 출력 형식을 구체적으로 지정
- 역할 기반 지시 ("You are a meeting secretary...")

**User Prompt 작성 팁:**
- 필요한 모든 정보를 나열
- 예시를 제공하면 더 좋은 결과
- 컨텍스트를 충분히 제공


```python
# 시스템 메시지: LLM의 역할과 출력 형식 정의
system_message = """
You produce minutes of meetings from transcripts, with summary, key discussion points,
takeaways and action items with owners, in markdown format without code blocks. 
"""

# 사용자 프롬프트: 구체적인 요구사항과 컨텍스트
user_prompt = f"""
Below is an extract transcript of a Denver council meeting.
Please write minutes in markdown without code blocks, including:
- a summary with attendees, location and date
- discussion points
- takeaways
- action items with owners
and korean translation.
Transcription:
{transcription}
"""

# OpenAI Chat API 형식으로 메시지 구성
messages = [
    {"role": "system", "content": system_message},  # LLM의 역할
    {"role": "user", "content": user_prompt}  # 사용자 요청
]

print("✓ 프롬프트 준비 완료!")
print(f"  - 시스템 메시지 길이: {len(system_message)} 문자")
print(f"  - 사용자 프롬프트 길이: {len(user_prompt)} 문자")
```



<div class="nb-output">

```text
✓ 프롬프트 준비 완료!
  - 시스템 메시지 길이: 169 문자
  - 사용자 프롬프트 길이: 9879 문자
```

</div>


## 3단계: 다양한 LLM으로 회의록 생성

이제 동일한 프롬프트를 사용하여 여러 LLM의 성능을 비교해보겠습니다.

### 옵션 1: Ollama (로컬 LLM)

**Ollama**는 로컬에서 LLM을 실행할 수 있는 오픈소스 도구입니다.

#### 장점
- 💰 **무료**: API 비용 없음
- 🔒 **프라이버시**: 데이터가 외부로 나가지 않음
- ⚡ **낮은 지연시간**: 네트워크 요청 불필요
- 🎯 **커스터마이징**: 모델 미세 조정 가능

#### 단점
- 💻 **하드웨어 요구사항**: GPU 필요 (8GB+ VRAM)
- 📊 **성능**: 클라우드 모델 대비 낮은 품질 가능
- 🔧 **설정 필요**: 초기 설치 및 구성


```


```python
# Ollama를 OpenAI API 형식으로 사용
client = OpenAI(
    base_url='http://localhost:11434/v1',  # Ollama 로컬 서버
    api_key='ollama'  # 더미 키 (형식 맞추기용)
)

# Llama 3.2로 회의록 생성
response = client.chat.completions.create(
    #model="llama3.2",  # Ollama에서 다운로드한 모델
    model="exaone3.5",  # Ollama에서 다운로드한 모델
    messages=[
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_prompt}
    ]
)

print("=== Ollama (Llama 3.2) 결과 ===")
display(Markdown(response.choices[0].message.content))
```



<div class="nb-output">

```text
=== Ollama (Llama 3.2) 결과 ===
<IPython.core.display.Markdown object>
```

</div>


### 옵션 2: OpenAI GPT-4 (클라우드 API)

**GPT-4**는 OpenAI의 최고 성능 모델입니다.

#### 장점
- 🎯 **최고 품질**: 가장 정확하고 자연스러운 출력
- 🚀 **즉시 사용**: 설치 불필요
- 🔧 **유지보수 불필요**: OpenAI가 관리
- 📈 **확장성**: 사용량에 따라 자동 확장

#### 단점
- 💰 **비용**: API 사용료 발생 (입력 $0.03/1K 토큰, 출력 $0.06/1K 토큰)
- 🌐 **네트워크 필요**: 인터넷 연결 필수
- 🔒 **프라이버시**: 데이터가 OpenAI 서버로 전송
- ⏱️ **레이트 리밋**: API 호출 제한 존재

#### 비용 예측
일반적인 회의록 (5분 음성):
- 입력: ~2,000 토큰 ($0.06)
- 출력: ~500 토큰 ($0.03)
- **총 비용**: ~$0.09/회의


```python
# OpenAI GPT-4 사용
client = OpenAI(api_key=openai_api_key)
response = client.chat.completions.create(
    model="gpt-4",  # 최고 성능 모델
    messages=[
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_prompt}
    ]
)

print("=== OpenAI GPT-4 결과 ===")
display(Markdown(response.choices[0].message.content))
```



<div class="nb-output">

```text
=== OpenAI GPT-4 결과 ===
<IPython.core.display.Markdown object>
```

</div>


### 모델 비교 요약

| 특징 | Ollama (로컬) | OpenAI GPT-4 |
|------|---------------|--------------|
| **비용** | 무료 (하드웨어 비용만) | ~$0.09/회의 |
| **품질** | 좋음 | 최고 |
| **속도** | 빠름 (로컬) | 보통 (네트워크) |
| **프라이버시** | 완전 보안 | 데이터 외부 전송 |
| **설정** | 복잡 | 간단 |
| **하드웨어** | GPU 필요 | 불필요 |
| **추천 용도** | 민감한 정보, 대량 처리 | 최고 품질 필요, 소량 처리 |

### 선택 기준

**Ollama를 선택하세요:**
- 민감한 정보를 다루는 경우
- 일일 100+ 회의록 생성 시
- 충분한 GPU 리소스 보유

**OpenAI를 선택하세요:**
- 최고 품질이 필요한 경우
- 소량의 회의록 (일 10개 미만)
- 빠른 시작이 중요한 경우


```python
# 긴 회의록을 청크로 나누기
def chunk_text(text, max_chars=10000, overlap=500):
    """
    텍스트를 겹침이 있는 청크로 분할
    
    Args:
        text: 분할할 텍스트
        max_chars: 청크당 최대 문자 수
        overlap: 청크 간 겹침 문자 수
    """
    chunks = []
    start = 0
    
    while start < len(text):
        end = start + max_chars
        chunks.append(text[start:end])
        start = end - overlap  # 겹침 적용
    
    return chunks

# 예제
long_text = transcription  # 실제 긴 회의록
chunks = chunk_text(long_text, max_chars=5000)

print(f"=== 청킹 결과 ===")
print(f"전체 길이: {len(long_text)} 문자")
print(f"청크 개수: {len(chunks)}")
print(f"각 청크 크기: {[len(c) for c in chunks]}")
```



<div class="nb-output">

```text
=== 청킹 결과 ===
전체 길이: 9678 문자
청크 개수: 3
각 청크 크기: [5000, 5000, 678]
```

</div>


### JSON 출력으로 구조화된 데이터 생성

Markdown 대신 JSON으로 출력하면 데이터베이스 저장이나 자동화가 쉬워집니다.

## 5단계: 실전 활용 팁

### 자동화 워크플로우

실무에서 회의록 자동화를 구축할 때 고려사항:

### 비용 최적화 전략

**1. 모델 선택 최적화**
```python
# 단계별로 다른 모델 사용
- 초안: GPT-3.5-turbo ($0.002/1K)
- 최종: GPT-4 ($0.03/1K)
- 대량: Ollama (무료)
```

**2. 프롬프트 최적화**
- 불필요한 컨텍스트 제거
- few-shot 예시 최소화
- 출력 길이 제한 설정

**3. 캐싱 활용**
- 유사한 회의는 템플릿 재사용
- 자주 사용하는 프롬프트 캐싱

### 품질 관리 체크리스트

✅ **필수 항목 확인:**
- [ ] 참석자 명단 완전성
- [ ] 날짜/시간/장소 정확성
- [ ] 액션 아이템 담당자 명시
- [ ] 결정 사항 명확성

✅ **일관성 검증:**
- [ ] 동일 형식 유지
- [ ] 전문 용어 일관성
- [ ] 시제 일관성

### 에러 핸들링

일반적인 문제와 해결책:

## 결론

### 핵심 요약

이 노트북에서 다룬 내용:

1. ✅ **Speech-to-Text**: OpenAI Whisper를 통한 음성 인식
2. ✅ **프롬프트 엔지니어링**: 효과적인 회의록 생성 프롬프트
3. ✅ **다양한 LLM**: Ollama(로컬) vs OpenAI(클라우드) 비교
4. ✅ **고급 기능**: 청킹, Map-Reduce, JSON 출력

### 실무 도입 로드맵

**Phase 1: MVP (1-2주)**
- OpenAI API로 기본 파이프라인 구축
- 소수의 회의에서 테스트
- 품질 검증 및 프롬프트 개선

**Phase 2: 최적화 (2-4주)**
- 비용 분석 및 모델 선택
- Ollama 도입 검토 (민감 정보용)
- 자동화 워크플로우 구축

**Phase 3: 확장 (1-2개월)**
- 전사 배포
- 다국어 지원
- 커스텀 템플릿 및 통합

### 예상 효과

📊 **정량적 효과:**
- ⏰ 회의록 작성 시간 **80% 감소**
- 💰 인건비 절감: 월 100회의 × 30분 = **50시간 절약**
- 📈 회의록 작성률 **95%+ 달성**

🎯 **정성적 효과:**
- 일관된 품질과 형식
- 빠른 공유 및 검색 가능
- 액션 아이템 추적 개선

### 다음 단계

**더 알아보기:**
- [OpenAI Whisper Documentation](https://platform.openai.com/docs/guides/speech-to-text)
- [Ollama Documentation](https://ollama.ai/docs)
- [LangChain for Production](https://python.langchain.com/)

**추가 개선 아이디어:**
- 실시간 회의록 생성 (스트리밍)
- 화자 분리 (Speaker Diarization)
- 감정 분석 추가
- Slack/Teams 통합
- 자동 이메일 발송

### 마지막 조언

> "완벽한 자동화보다 90% 자동화 + 10% 인간 검토가 더 실용적입니다."

AI가 초안을 생성하고, 사람이 최종 검토하는 하이브리드 접근이 최적의 결과를 만듭니다.

---

**Happy Automating!** 🚀

이 노트북이 여러분의 회의록 작성을 혁신하는 데 도움이 되기를 바랍니다!


```python
# 견고한 에러 핸들링
import time
from openai import OpenAI, OpenAIError

def robust_meeting_minutes(audio_path, max_retries=3):
    """재시도 로직이 포함된 안정적인 파이프라인"""
    
    for attempt in range(max_retries):
        try:
            # Speech-to-Text
            with open(audio_path, "rb") as audio_file:
                transcription = openai.audio.transcriptions.create(
                    model="gpt-4o-mini-transcribe",
                    file=audio_file,
                    response_format="text"
                )
            
            # 회의록 생성
            client = OpenAI(api_key=openai_api_key)
            response = client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": f"Generate minutes: {transcription}"}
                ],
                timeout=60  # 타임아웃 설정
            )
            
            return response.choices[0].message.content
            
        except OpenAIError as e:
            print(f"시도 {attempt + 1}/{max_retries} 실패: {e}")
            
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt  # 지수 백오프
                print(f"{wait_time}초 후 재시도...")
                time.sleep(wait_time)
            else:
                print("최대 재시도 횟수 초과")
                raise
        
        except FileNotFoundError:
            print(f"❌ 오디오 파일을 찾을 수 없습니다: {audio_path}")
            return None
        
        except Exception as e:
            print(f"❌ 예상치 못한 오류: {e}")
            return None

print("✓ 에러 핸들링 함수 준비 완료")
```



```python
# 완전한 파이프라인 함수
def meeting_minutes_pipeline(audio_path, output_format="markdown"):
    """
    음성 파일부터 회의록까지 전체 파이프라인
    
    Args:
        audio_path: 오디오 파일 경로
        output_format: "markdown" 또는 "json"
    
    Returns:
        회의록 (문자열 또는 딕셔너리)
    """
    print("1단계: 음성 → 텍스트 변환...")
    
    # Speech-to-Text
    with open(audio_path, "rb") as audio_file:
        transcription = openai.audio.transcriptions.create(
            model="gpt-4o-mini-transcribe",
            file=audio_file,
            response_format="text"
        )
    
    print(f"   ✓ 변환 완료 ({len(transcription)} 문자)")
    
    # 2단계: 회의록 생성
    print("2단계: 회의록 생성...")
    
    if output_format == "json":
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": json_system_message},
                {"role": "user", "content": f"Generate minutes: {transcription}"}
            ],
            response_format={"type": "json_object"}
        )
        result = json.loads(response.choices[0].message.content)
    else:
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": f"Generate minutes: {transcription}"}
            ]
        )
        result = response.choices[0].message.content
    
    print("   ✓ 회의록 생성 완료!")
    return result

# 사용 예시
# minutes = meeting_minutes_pipeline("./denver_extract.mp3", output_format="json")
# print(json.dumps(minutes, indent=2, ensure_ascii=False))
```



```python
import json

# JSON 출력을 위한 프롬프트
json_system_message = """
You are a meeting minutes generator. Output ONLY valid JSON with this structure:
{
  "meeting_info": {
    "date": "YYYY-MM-DD",
    "location": "string",
    "attendees": ["name1", "name2"]
  },
  "summary": "string",
  "discussion_points": ["point1", "point2"],
  "decisions": ["decision1", "decision2"],
  "action_items": [
    {"task": "string", "owner": "string", "deadline": "string"}
  ]
}
"""

json_prompt = f"Generate meeting minutes as JSON from this transcript:\n\n{transcription[:2000]}"

# GPT-4로 JSON 생성
client = OpenAI(api_key=openai_api_key)
response = client.chat.completions.create(
    model="gpt-4",
    messages=[
        {"role": "system", "content": json_system_message},
        {"role": "user", "content": json_prompt}
    ],
    response_format={"type": "json_object"}  # JSON 모드 강제
)

# 결과 파싱
minutes_json = json.loads(response.choices[0].message.content)

print("=== JSON 형식 회의록 ===")
print(json.dumps(minutes_json, indent=2, ensure_ascii=False))
```



```python
# Map-Reduce 패턴으로 청크 처리
def summarize_chunks(chunks, client, model="gpt-4"):
    """각 청크를 요약한 후 최종 요약 생성"""
    
    # 1단계: 각 청크 요약 (Map)
    chunk_summaries = []
    for i, chunk in enumerate(chunks):
        print(f"청크 {i+1}/{len(chunks)} 처리 중...")
        
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "Summarize this meeting transcript segment."},
                {"role": "user", "content": chunk}
            ]
        )
        chunk_summaries.append(response.choices[0].message.content)
    
    # 2단계: 청크 요약들을 결합하여 최종 요약 (Reduce)
    combined = "\n\n".join(chunk_summaries)
    
    final_response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_message},
            {"role": "user", "content": f"Create final minutes from these summaries:\n\n{combined}"}
        ]
    )
    
    return final_response.choices[0].message.content

# 사용 예시 (주석 처리 - 실행 시 비용 발생)
# client = OpenAI(api_key=openai_api_key)
# final_minutes = summarize_chunks(chunks, client)
# display(Markdown(final_minutes))
```


## 4단계: 고급 기능

### 긴 회의록 처리 (청킹 전략)

대부분의 LLM은 컨텍스트 윈도우 제한이 있습니다. 긴 회의는 여러 청크로 나누어 처리해야 합니다.
