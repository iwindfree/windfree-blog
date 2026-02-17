---
title: "ASM을 이용한 Java 클래스 조작 (Part 3)"
author: iwindfree
pubDatetime: 2019-10-19T11:00:00Z
slug: "java-bci-asm-part3"
category: "JAVA BCI"
series: "JAVA BCI"
seriesOrder: 3
tags: ["java", "bci", "asm"]
description: "JVM의 실행 모델인 Stack Frame(Local Variable Array, Operand Stack)의 동작 원리와 JVM 명령어 카테고리, ClassWriter 옵션을 설명합니다."
canonicalURL: "https://iwindfree.wordpress.com/2019/10/19/manuplating-java-class-with-asm-part3/"
---

## JVM 실행 모델

[Part 2](/posts/java-bci-asm-part2)에서는 ClassVisitor 를 활용하여 클래스를 변형하는 방법을 다뤘습니다. 다음 단계인 **메서드 변형**으로 넘어가기 전에, JVM 이 메서드를 어떻게 실행하는지 기본 모델을 먼저 이해해 두면 도움이 됩니다.

## Java Virtual Machine Stack & Stack Frame

> **한 줄 요약**: 쓰레드마다 Stack 이 하나씩 있고, 메서드가 호출될 때마다 Stack Frame 이 하나씩 쌓입니다.

Java 코드는 쓰레드 내에서 실행되며, 각 쓰레드는 자신만의 `Java Virtual Machine Stack` 을 갖고 있습니다. 이 Stack 은 `Stack Frame` 들로 구성되어 있습니다. 책을 쌓듯이, 메서드가 호출될 때마다 새로운 Stack Frame 이 위에 쌓이는 구조입니다.

각 `Stack Frame` 은 메서드 호출 한 건을 나타냅니다. 메서드가 호출되면 새로운 Stack Frame 이 생성되어 현재 쓰레드의 `Java Virtual Machine Stack` 에 push 됩니다. 메서드가 종료되면 해당 Stack Frame 은 pop 되어 제거되고, 호출한 쪽 메서드에서 실행이 계속됩니다.

## Stack Frame 구성 요소

Stack Frame 은 크게 두 부분으로 구성됩니다: **Local Variable Array** 와 **Operand Stack** 입니다.

### Local Variable Array

> **한 줄 요약**: 메서드의 파라미터와 지역 변수를 인덱스로 접근하는 배열입니다. 인스턴스 메서드라면 index 0 은 항상 `this` 입니다.

Local Variable Array 는 인덱스를 통해 접근할 수 있는 변수 저장소입니다. `변수 테이블(variable table)` 이라고도 부르며, 메서드의 파라미터와 지역 변수에 대한 정보를 담고 있습니다.

해당 Stack Frame 이 인스턴스 메서드나 생성자의 것이라면, 배열의 첫 번째 요소(index 0)에는 `this` 에 대한 참조가 저장됩니다. `this` 가 index 0 을 차지하는 이유는, 인스턴스 메서드가 항상 자기 자신의 객체를 알고 있어야 필드에 접근하거나 다른 인스턴스 메서드를 호출할 수 있기 때문입니다. index 1 부터는 메서드의 파라미터가 선언 순서대로 저장됩니다.

메서드의 파라미터는 선언된 순서로 인덱스가 할당되며, 지역 변수는 컴파일러가 임의로 인덱스를 할당합니다.

변수의 타입에 따라 저장 방식이 달라집니다:
- **primitive 타입** (`int`, `long`, `float` 등) — 변수의 **값 자체**가 Local Variable Array 에 저장됩니다.
- **reference 타입** (객체, 배열 등) — 실제 객체가 저장된 **heap 의 주소값**이 저장됩니다.

### Operand Stack

> **한 줄 요약**: JVM 이 연산을 수행할 때 사용하는 임시 작업 공간입니다.

Operand Stack 은 일종의 JVM 의 작업 공간입니다. 다양한 JVM 명령어가 사용하는 인자의 값이나 리턴값을 주고받는 공간입니다.

예를 들어 `iadd` 라는 JVM 명령어는 Operand Stack 에서 두 개의 정수값을 꺼낸 후(pop), 그 두 수를 더한 결과값을 다시 Operand Stack 에 push 합니다.

## 예제 코드

위에서 설명한 Local Variable Array 와 Operand Stack 이 실제로 어떻게 동작하는지, 간단한 덧셈 예제를 통해 확인해 보겠습니다.

```java
class Test {
    public void method() {
        int i, j, k;
        i = 8;
        j = 6;
        k = i + j;
    }
}
```

위 코드를 컴파일하면 다음과 같은 바이트코드가 생성됩니다:

```
bipush 8
istore_1
bipush 6
istore_2
iload_1
iload_2
iadd
istore_3
return
```

각 명령의 동작:

1. `bipush 8` : 8 이라는 상수를 Operand Stack 에 push 합니다.
2. `istore_1` : Operand Stack 에서 값을 pop 하여 Local Variable Array 의 1번 인덱스에 저장합니다. (변수 `i`)
3. `bipush 6` : 6 이라는 상수를 Operand Stack 에 push 합니다.
4. `istore_2` : Operand Stack 에서 값을 pop 하여 Local Variable Array 의 2번 인덱스에 저장합니다. (변수 `j`)
5. `iload_1` : Local Variable Array 1번 인덱스에서 값을 load 하여 Operand Stack 에 push 합니다.
6. `iload_2` : Local Variable Array 2번 인덱스에서 값을 load 하여 Operand Stack 에 push 합니다.
7. `iadd` : Operand Stack 에서 두 값을 pop 하여 더하고, 결과값을 다시 Operand Stack 에 push 합니다.
8. `istore_3` : Operand Stack 에서 결과값을 pop 하여 Local Variable Array 3번 인덱스에 저장합니다. (변수 `k`)
9. `return` : 메서드를 종료합니다.

아래 테이블은 각 명령어가 실행될 때마다 Local Variable Array 와 Operand Stack 의 상태가 어떻게 변하는지 보여줍니다. `method()` 는 인스턴스 메서드이므로 index 0 에 `this` 가 저장되어 있는 것을 확인할 수 있습니다.

| 단계 | 명령어 | Operand Stack | Local Variable Array |
|------|--------|---------------|----------------------|
| 0 | (초기) | `[]` | `[this, -, -, -]` |
| 1 | `bipush 8` | `[8]` | `[this, -, -, -]` |
| 2 | `istore_1` | `[]` | `[this, 8, -, -]` |
| 3 | `bipush 6` | `[6]` | `[this, 8, -, -]` |
| 4 | `istore_2` | `[]` | `[this, 8, 6, -]` |
| 5 | `iload_1` | `[8]` | `[this, 8, 6, -]` |
| 6 | `iload_2` | `[8, 6]` | `[this, 8, 6, -]` |
| 7 | `iadd` | `[14]` | `[this, 8, 6, -]` |
| 8 | `istore_3` | `[]` | `[this, 8, 6, 14]` |
| 9 | `return` | `[]` | `[this, 8, 6, 14]` |

## JVM 명령어 카테고리

JVM 의 주요 명령어를 카테고리별로 정리하면 다음과 같습니다.

| 카테고리 | 명령어 | 설명 |
|---------|--------|------|
| **STACK** | `POP`, `DUP`, `SWAP` | `POP` 은 스택 꼭대기 값을 제거하고, `DUP` 은 꼭대기 값의 복사본을 push 하고, `SWAP` 은 꼭대기 두 값의 순서를 바꿉니다. |
| **CONSTANTS** | `ACONST_NULL`, `ICONST_0`, `FCONST_0`, `DCONST_0`, `BIPUSH`, `SIPUSH`, `LDC` | `ACONST_NULL` 은 null 을 push 하고, `ICONST_0` 은 int 값 0 을 push 합니다. `LDC` 는 int, float, long, double, String, class 등 임의의 상수를 push 합니다. |
| **ARITHMETIC & LOGIC** | `xADD`, `xSUB`, `xMUL`, `xDIV`, `xREM` | x 는 I, L, F, D 중 하나입니다. 마찬가지로 `<<`, `>>`, `>>>`, `\|`, `^` 에 해당하는 int/long 전용 명령어도 있습니다. |
| **CAST** | `I2F`, `F2D`, `L2D`, `CHECKCAST` | 숫자 값을 다른 숫자 타입으로 변환합니다. `CHECKCAST t` 는 참조 값을 타입 t 로 캐스팅합니다. |
| **OBJECTS** | `NEW` | `NEW type` 명령어는 지정한 type 의 새 객체를 생성하여 스택에 push 합니다. |
| **FIELDS** | `GETFIELD`, `PUTFIELD`, `GETSTATIC`, `PUTSTATIC` | `GETFIELD` 는 객체 참조를 pop 한 뒤 해당 필드의 값을 push 합니다. `GETSTATIC` 과 `PUTSTATIC` 은 정적 필드에 대한 유사한 명령어입니다. |
| **METHODS** | `INVOKEVIRTUAL`, `INVOKESTATIC`, `INVOKESPECIAL`, `INVOKEINTERFACE`, `INVOKEDYNAMIC` | `INVOKEVIRTUAL` 은 인스턴스 메서드, `INVOKESTATIC` 은 정적 메서드, `INVOKESPECIAL` 은 private 메서드와 생성자를 호출합니다. |
| **ARRAYS** | `xALOAD`, `xASTORE` | x 는 I, L, F, D, A 외에도 B, C, S 가 될 수 있습니다. |
| **JUMPS** | `IFEQ`, `IFNE`, `IFGE`, `TABLESWITCH`, `LOOKUPSWITCH` | `IFEQ` 는 int 값을 pop 하여 0 이면 점프합니다. `TABLESWITCH` 와 `LOOKUPSWITCH` 는 Java 의 switch 문에 해당합니다. |
| **RETURN** | `RETURN`, `xRETURN` | `RETURN` 은 void 메서드에서, `xRETURN` 은 값을 반환하는 메서드에서 사용됩니다. |

## ClassWriter 의 옵션

> **한 줄 요약**: 바이트코드를 수정하면 Stack Frame 크기를 다시 계산해야 하는데, `COMPUTE_FRAMES` 를 사용하면 ASM 이 전부 자동으로 처리해 줍니다.

바이트코드를 수정하면 Local Variable Array 나 Operand Stack 의 크기가 달라질 수 있으므로, Stack Frame 의 크기도 함께 조정해야 합니다. ClassWriter 는 이를 위해 세 가지 옵션을 지원합니다.

- `new ClassWriter(0)` : 아무 것도 자동으로 계산하지 않습니다. 프레임, 로컬 변수 및 피연산자 스택 크기를 직접 계산해야 합니다. 바이트코드를 정밀하게 제어하고 싶을 때 사용합니다.
- `new ClassWriter(ClassWriter.COMPUTE_MAXS)` : 로컬 변수와 피연산자 스택의 최대 크기를 자동으로 계산합니다. 다만 Stack Map Frame 은 직접 관리해야 합니다.
- `new ClassWriter(ClassWriter.COMPUTE_FRAMES)` : 모든 것을 자동으로 계산합니다. 로컬 변수, 피연산자 스택 크기뿐 아니라 Stack Map Frame 까지 자동으로 처리해 줍니다.

대부분의 경우 `COMPUTE_FRAMES` 를 사용하는 것이 가장 편리합니다. Part 2 의 ClassTransformer 예제에서도 `new ClassWriter(cr, ClassWriter.COMPUTE_FRAMES)` 를 사용한 것을 확인할 수 있습니다. 수동 계산은 오류가 나기 쉽고, `COMPUTE_FRAMES` 가 성능상 큰 차이를 만들지 않으므로 특별한 이유가 없다면 이 옵션을 권장합니다.

---

다음 [Part 4](/posts/java-bci-asm-part4)에서는 이 실행 모델 지식을 바탕으로, MethodVisitor 를 사용하여 실제로 메서드를 변형하는 방법을 다룹니다.
