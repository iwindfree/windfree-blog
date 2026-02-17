---
title: "08-1. 데이터셋 개념 정리"
author: iwindfree
pubDatetime: 2025-02-24T09:00:00Z
slug: "llm-fine-tuning-dataset"
category: "LLM Engineering"
series: "LLM Engineering"
seriesOrder: 21
tags: ["ai", "llm", "fine-tuning"]
description: "머신러닝/딥러닝 모델을 학습시킬 때 데이터를 어떻게 나누고 활용하는지 알아봅니다."
---

머신러닝/딥러닝 모델을 학습시킬 때 데이터를 어떻게 나누고 활용하는지 알아봅니다.

## 1. 데이터셋 분할의 필요성

모델을 학습시킬 때 **모든 데이터를 학습에 사용하면 안 됩니다**. 왜일까요?

- 모델이 학습 데이터를 "암기"할 수 있음 (과적합, Overfitting)
- 새로운 데이터에 대한 성능을 측정할 수 없음
- 하이퍼파라미터 튜닝 시 객관적인 기준이 없음

이를 해결하기 위해 데이터를 **Training, Validation, Test** 세 가지로 나눕니다.

## 2. 세 가지 데이터셋

```
┌─────────────────────────────────────────────────────────────────┐
│                        전체 데이터셋                              │
├───────────────────────────┬─────────────────┬───────────────────┤
│      Training Set         │ Validation Set  │     Test Set      │
│        (60-80%)           │    (10-20%)     │     (10-20%)      │
│                           │                 │                   │
│   모델 학습에 사용          │  하이퍼파라미터   │   최종 성능 평가    │
│   (가중치 업데이트)         │  튜닝에 사용     │   (한 번만 사용)    │
└───────────────────────────┴─────────────────┴───────────────────┘
```

### 2.1 Training Set (훈련 데이터)

**목적**: 모델의 파라미터(가중치)를 학습시키는 데 사용

**특징**:
- 전체 데이터의 60-80%를 차지
- 모델이 패턴을 학습하는 데 직접 사용됨
- 여러 번 반복(epoch)하여 학습

**비유**: 학생이 공부하는 교과서와 문제집

### 2.2 Validation Set (검증 데이터)

**목적**: 학습 중 모델의 성능을 모니터링하고 하이퍼파라미터를 조정

**특징**:
- 전체 데이터의 10-20%를 차지
- 학습에 직접 사용되지 않음 (가중치 업데이트 X)
- 학습 과정에서 여러 번 평가에 사용
- Early Stopping, 모델 선택 등에 활용

**비유**: 학생이 공부 중간중간 푸는 모의고사

**주요 용도**:
- 과적합(Overfitting) 감지
- 하이퍼파라미터 튜닝 (학습률, 배치 크기, 레이어 수 등)
- 최적의 학습 시점(epoch) 결정

### 2.3 Test Set (테스트 데이터)

**목적**: 최종 모델의 성능을 객관적으로 평가

**특징**:
- 전체 데이터의 10-20%를 차지
- **단 한 번만** 사용 (모델 개발 완료 후)
- 모델이 한 번도 보지 못한 데이터
- 실제 서비스 환경에서의 성능을 추정

**비유**: 학생이 치르는 실제 수능 시험

**주의사항**:
- Test Set으로 하이퍼파라미터를 튜닝하면 안 됨
- Test Set 성능을 보고 모델을 수정하면 안 됨
- 그렇게 하면 Test Set도 간접적으로 학습에 사용된 것

## 3. 과적합(Overfitting)과 데이터셋의 관계

```
손실(Loss)
    │
    │   \                          
    │    \    Training Loss        
    │     \__________________      
    │      \                       
    │       \   Validation Loss    
    │        \____                 
    │             \___/‾‾‾‾‾‾‾‾    ← 과적합 시작점
    │                              
    └──────────────────────────── Epoch
          ↑
     최적 중단점 (Early Stopping)
```

- **Training Loss**는 계속 감소
- **Validation Loss**는 어느 시점부터 증가 → 과적합 신호
- Validation Loss가 증가하기 시작하면 학습 중단 (Early Stopping)

## 4. 실제 분할 비율 예시

| 데이터 크기 | Training | Validation | Test | 비고 |
|------------|----------|------------|------|------|
| 소규모 (< 1만) | 60% | 20% | 20% | 검증 데이터 충분히 확보 |
| 중규모 (1-10만) | 70% | 15% | 15% | 일반적인 비율 |
| 대규모 (> 100만) | 98% | 1% | 1% | 1%도 충분한 샘플 수 |

**핵심**: Validation/Test Set은 통계적으로 유의미한 크기면 충분

## 5. 코드 예시: 데이터셋 분할


```python
from sklearn.model_selection import train_test_split
import numpy as np

# 예시 데이터 생성
X = np.random.randn(1000, 10)  # 1000개 샘플, 10개 특성
y = np.random.randint(0, 2, 1000)  # 이진 분류 레이블

# 1단계: Train+Val / Test 분할 (80% / 20%)
X_train_val, X_test, y_train_val, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 2단계: Train / Val 분할 (Train+Val의 75% / 25% → 전체의 60% / 20%)
X_train, X_val, y_train, y_val = train_test_split(
    X_train_val, y_train_val, test_size=0.25, random_state=42
)

print(f"Training Set:   {len(X_train)} samples ({len(X_train)/len(X)*100:.0f}%)")
print(f"Validation Set: {len(X_val)} samples ({len(X_val)/len(X)*100:.0f}%)")
print(f"Test Set:       {len(X_test)} samples ({len(X_test)/len(X)*100:.0f}%)")
```


## 6. LLM Fine-tuning에서의 데이터셋

LLM을 파인튜닝할 때도 동일한 원칙이 적용됩니다.

### OpenAI Fine-tuning 예시

```python
# OpenAI는 Training과 Validation 파일을 별도로 업로드
training_file = client.files.create(
    file=open("train.jsonl", "rb"),
    purpose="fine-tune"
)

validation_file = client.files.create(
    file=open("validation.jsonl", "rb"),
    purpose="fine-tune"
)

# Fine-tuning 작업 생성
job = client.fine_tuning.jobs.create(
    training_file=training_file.id,
    validation_file=validation_file.id,  # 선택사항이지만 권장
    model="gpt-4o-mini-2024-07-18"
)
```

### JSONL 형식 예시

```json
{"messages": [{"role": "system", "content": "You are a helpful assistant."}, {"role": "user", "content": "Hello"}, {"role": "assistant", "content": "Hi there!"}]}
{"messages": [{"role": "system", "content": "You are a helpful assistant."}, {"role": "user", "content": "What is 2+2?"}, {"role": "assistant", "content": "4"}]}
```

## 7. 요약

| 데이터셋 | 목적 | 사용 시점 | 사용 횟수 |
|---------|------|----------|----------|
| **Training** | 모델 학습 (가중치 업데이트) | 학습 중 | 여러 번 (매 epoch) |
| **Validation** | 하이퍼파라미터 튜닝, 과적합 감지 | 학습 중 | 여러 번 |
| **Test** | 최종 성능 평가 | 학습 완료 후 | **단 한 번** |

### 핵심 원칙

1. **Test Set은 금고에 넣어두세요** - 최종 평가 전까지 절대 보지 않기
2. **Validation Set으로 튜닝하세요** - 하이퍼파라미터 조정은 여기서
3. **분할은 무작위로** - 데이터 편향 방지를 위해 shuffle 후 분할
4. **재현성 확보** - random_state 설정으로 동일한 분할 보장

## 8. 추가 개념: K-Fold Cross Validation

데이터가 적을 때 더 신뢰할 수 있는 검증을 위해 K-Fold 교차 검증을 사용합니다.

```
5-Fold Cross Validation 예시:

Fold 1: [Val] [Train] [Train] [Train] [Train]
Fold 2: [Train] [Val] [Train] [Train] [Train]
Fold 3: [Train] [Train] [Val] [Train] [Train]
Fold 4: [Train] [Train] [Train] [Val] [Train]
Fold 5: [Train] [Train] [Train] [Train] [Val]

→ 5번 학습하고 평균 성능 계산
```

- 모든 데이터가 한 번씩 검증에 사용됨
- 더 안정적인 성능 추정 가능
- 단점: 학습 시간이 K배 증가


```python
from sklearn.model_selection import KFold
import numpy as np

# 예시 데이터
X = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]])
y = np.array([0, 1, 0, 1, 0])

# 5-Fold Cross Validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)

for fold, (train_idx, val_idx) in enumerate(kf.split(X), 1):
    print(f"Fold {fold}:")
    print(f"  Train indices: {train_idx}")
    print(f"  Val indices:   {val_idx}")
```

