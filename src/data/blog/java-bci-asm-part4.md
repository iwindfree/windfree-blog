---
title: "ASM을 이용한 Java 클래스 조작 (Part 4)"
author: iwindfree
pubDatetime: 2019-10-19T12:00:00Z
slug: "java-bci-asm-part4"
category: "JAVA BCI"
series: "JAVA BCI"
seriesOrder: 4
tags: ["java", "bci", "asm"]
description: "MethodVisitor를 사용하여 메서드를 변형하고, 메서드 실행 시간을 측정하는 기능을 소스 코드 수정 없이 추가하는 예제를 구현합니다."
canonicalURL: "https://iwindfree.wordpress.com/2019/10/19/manuplating-java-class-with-asm-part4/"
---

## 메서드 변형하기

[Part 3](/posts/java-bci-asm-part3)에서는 JVM 의 실행 모델 — Stack Frame, Local Variable Array, Operand Stack — 을 살펴보았습니다. 이번 Part 4에서는 그 지식을 바탕으로, **MethodVisitor 를 사용하여 실제로 메서드를 변형하는 방법**을 다룹니다. 최종 예제로, 소스 코드 수정 없이 메서드의 실행 시간을 측정하는 기능을 추가해 보겠습니다.

## MethodVisitor 클래스

> **한 줄 요약**: ClassVisitor 가 클래스 전체를 다루는 Visitor 라면, MethodVisitor 는 **메서드 하나**를 다루는 Visitor 입니다. 위임 체인 원리도 동일합니다.

클래스의 메서드를 생성하거나 변형하는 ASM 의 API 는 `MethodVisitor` 라는 추상 클래스에 기반을 두고 있습니다. MethodVisitor 는 [Part 2](/posts/java-bci-asm-part2)에서 살펴보았던 ClassVisitor 의 `visitMethod` 메서드의 리턴값이기도 합니다.

[Part 2](/posts/java-bci-asm-part2)에서 ClassVisitor 의 위임 체인을 설명했었는데, MethodVisitor 도 동일한 원리로 동작합니다. ClassVisitor 에서 override 하지 않은 `visitXXX` 메서드가 다음 ClassVisitor 에게 위임되었듯이, MethodVisitor 에서도 override 하지 않은 메서드는 생성자에서 전달받은 다음 MethodVisitor 에게 위임됩니다.

MethodVisitor 의 메서드 호출 순서는 다음과 같습니다:

```
visitAnnotationDefault?
( visitAnnotation | visitParameterAnnotation | visitAttribute )*
( visitCode
  ( visitTryCatchBlock | visitLabel | visitFrame | visitXxxInsn |
    visitLocalVariable | visitLineNumber )*
  visitMaxs )?
visitEnd
```

`visitCode` 부터 `visitMaxs` 까지가 메서드의 바이트코드를 다루는 구간입니다. 이 구간 안에서 `visitXxxInsn` 메서드들을 통해 바이트코드 명령어를 하나씩 방문하게 됩니다.

### 주요 메서드

| 함수명 | 설명 |
|--------|------|
| `visitCode` | 메서드의 바이트코드 시작점을 읽을 때 호출됩니다. 메서드 **앞부분에 코드를 삽입**하고 싶을 때 이 메서드를 override 합니다. |
| `visitInsn` | 피연산자가 없는 명령어(`RETURN`, `LSUB`, `LADD`, `DUP` 등)를 만날 때 호출됩니다. Part 3에서 다룬 STACK, ARITHMETIC, RETURN 카테고리의 명령어가 여기에 해당합니다. |
| `visitFieldInsn` | 필드의 값을 저장하거나 불러올 때 호출됩니다. Part 3의 FIELDS 카테고리(`GETSTATIC`, `PUTSTATIC` 등)에 해당합니다. |
| `visitMethodInsn` | 메서드를 호출하는 명령어를 만날 때 호출됩니다. Part 3의 METHODS 카테고리(`INVOKEVIRTUAL`, `INVOKESTATIC` 등)에 해당합니다. |
| `visitMaxs` | 메서드의 Stack Frame 에서 Operand Stack 의 최대 크기와 Local Variable Array 의 최대 크기를 설정할 때 호출됩니다. |
| `visitEnd` | 메서드의 바이트코드 끝을 읽을 때 호출됩니다. |

## 예제: 메서드 응답시간 측정하기

지금까지 배운 내용을 바탕으로, 실전 예제를 구현해 보겠습니다. 클래스에 있는 특정 메서드의 수행 시간을 `System.out` 으로 출력하는 기능을, **소스 코드 수정 없이** 추가합니다.

이 예제는 다음 4개 클래스로 구성됩니다:

1. **AddFieldCV** — `static long timer` 필드를 추가합니다 ([Part 2](/posts/java-bci-asm-part2)에서 구현)
2. **ElapsedTimeCV** — 특정 메서드를 만나면 ElapsedTimeMV 로 위임합니다
3. **ElapsedTimeMV** — 메서드 시작/종료 지점에 시간 측정 바이트코드를 삽입합니다
4. **ClassTransformer** — 위 클래스들을 변환 체인으로 조립합니다

변환 체인은 `ClassReader → ElapsedTimeCV → AddFieldCV → ClassWriter` 순서로 구성됩니다. ClassReader 가 클래스를 읽으면서 이벤트를 발생시키면, ElapsedTimeCV 가 먼저 받아서 메서드 변형을 처리하고, AddFieldCV 가 `timer` 필드를 추가하고, 최종적으로 ClassWriter 가 바이트코드를 생성합니다.

### 원본 코드

**AppTest.java:**

```java
public class AppTest {
    public void testMethod1() {
        System.out.println("Start Method1");
        try {
            Thread.sleep(1000);
        } catch (InterruptedException e) {}
        System.out.println("End Method1");
    }
}
```

**AppMain.java:**

```java
public class AppMain {
    public static void main(String[] args) {
        System.out.println("App start...");
        AppTest t = new AppTest();
        t.testMethod1();
    }
}
```

### static 변수 timer 추가

[Part 2](/posts/java-bci-asm-part2)에서 구현한 `AddFieldCV` 를 그대로 사용합니다. `AddFieldCV` 는 클래스에 같은 이름의 필드가 없을 때만 새 필드를 추가하는 ClassVisitor 였습니다. 여기서는 `static long timer` 필드를 추가하는 데 사용합니다. 자세한 구현 내용은 [Part 2](/posts/java-bci-asm-part2)를 참고해 주세요.

### 메서드 변형 — ElapsedTimeCV

> **한 줄 요약**: 변형 대상 메서드를 만나면 ElapsedTimeMV 로 위임하고, 나머지 메서드는 그대로 통과시키는 ClassVisitor 입니다.

```java
import org.objectweb.asm.ClassVisitor;
import org.objectweb.asm.MethodVisitor;

public class ElapsedTimeCV extends ClassVisitor {

    public ElapsedTimeCV(int api, ClassVisitor cv) {
        super(api, cv);
    }

    @Override
    public MethodVisitor visitMethod(int access, String name, String desc,
                                      String signature, String[] exceptions) {
        if (name.equals("testMethod1")) {
            MethodVisitor mv = super.visitMethod(access, name, desc,
                signature, exceptions);
            return new ElapsedTimeMV(api, mv);
        } else {
            return cv.visitMethod(access, name, desc, signature, exceptions);
        }
    }
}
```

`visitMethod` 는 ClassReader 가 메서드를 만날 때마다 호출됩니다. 여기서 메서드 이름이 `testMethod1` 인지 확인하여 분기합니다:

- **대상 메서드인 경우**: `super.visitMethod(...)` 를 호출하여 원래의 MethodVisitor 를 얻은 뒤, 이것을 `ElapsedTimeMV` 로 감싸서 반환합니다. 이렇게 하면 해당 메서드의 바이트코드를 방문할 때 ElapsedTimeMV 의 메서드들이 호출되어, 시간 측정 코드를 삽입할 수 있습니다.
- **대상이 아닌 경우**: `cv.visitMethod(...)` 를 호출하여 다음 ClassVisitor(AddFieldCV)에게 그대로 위임합니다. 변형 없이 통과시키는 것입니다.

### 시간 측정 코드 삽입 — ElapsedTimeMV

> **한 줄 요약**: `timer = timer - startTime + endTime` 이라는 트릭으로 경과 시간을 구합니다. 필드 하나만으로 시작/종료 시간을 처리할 수 있습니다.

이 클래스가 이 글의 **핵심**입니다. 메서드 시작 시점과 종료 시점에 바이트코드를 삽입하여 실행 시간을 측정합니다.

#### timer 트릭의 원리

별도의 지역 변수 없이 `static long timer` 필드 하나만으로 경과 시간을 구하는 기법입니다:

- 메서드 시작 시: `timer = timer - System.currentTimeMillis()`
- 메서드 종료 시: `timer = timer + System.currentTimeMillis()`

`timer` 의 초기값이 0이라고 가정하면:

1. 시작 (현재 시각이 1000ms 라고 가정): `timer = 0 - 1000 = -1000`
2. 종료 (현재 시각이 2000ms 라고 가정): `timer = -1000 + 2000 = 1000`

결과적으로 `timer` 에는 `종료 시각 - 시작 시각 = 경과 시간` 이 저장됩니다.

#### ElapsedTimeMV 전체 코드

```java
import org.objectweb.asm.MethodVisitor;
import org.objectweb.asm.Opcodes;

public class ElapsedTimeMV extends MethodVisitor {

    public ElapsedTimeMV(int api, MethodVisitor mv) {
        super(api, mv);
    }

    @Override
    public void visitCode() {
        mv.visitCode();
        mv.visitFieldInsn(Opcodes.GETSTATIC, "testapp1/AppTest", "timer", "J");
        mv.visitMethodInsn(Opcodes.INVOKESTATIC, "java/lang/System",
            "currentTimeMillis", "()J", false);
        mv.visitInsn(Opcodes.LSUB);
        mv.visitFieldInsn(Opcodes.PUTSTATIC, "testapp1/AppTest", "timer", "J");
    }

    @Override
    public void visitInsn(int opcode) {
        if ((opcode >= Opcodes.IRETURN && opcode <= Opcodes.RETURN)
                || opcode == Opcodes.ATHROW) {
            mv.visitFieldInsn(Opcodes.GETSTATIC, "testapp1/AppTest",
                "timer", "J");
            mv.visitMethodInsn(Opcodes.INVOKESTATIC, "java/lang/System",
                "currentTimeMillis", "()J", false);
            mv.visitInsn(Opcodes.LADD);
            mv.visitFieldInsn(Opcodes.PUTSTATIC, "testapp1/AppTest",
                "timer", "J");

            mv.visitFieldInsn(Opcodes.GETSTATIC, "java/lang/System",
                "out", "Ljava/io/PrintStream;");
            mv.visitTypeInsn(Opcodes.NEW, "java/lang/StringBuilder");
            mv.visitInsn(Opcodes.DUP);
            mv.visitLdcInsn("elapsed time: ");
            mv.visitMethodInsn(Opcodes.INVOKESPECIAL,
                "java/lang/StringBuilder", "<init>",
                "(Ljava/lang/String;)V", false);
            mv.visitFieldInsn(Opcodes.GETSTATIC, "testapp1/AppTest",
                "timer", "J");
            mv.visitMethodInsn(Opcodes.INVOKEVIRTUAL,
                "java/lang/StringBuilder", "append",
                "(J)Ljava/lang/StringBuilder;", false);
            mv.visitMethodInsn(Opcodes.INVOKEVIRTUAL,
                "java/lang/StringBuilder", "toString",
                "()Ljava/lang/String;", false);
            mv.visitMethodInsn(Opcodes.INVOKEVIRTUAL,
                "java/io/PrintStream", "println",
                "(Ljava/lang/String;)V", false);
        }
        mv.visitInsn(opcode);
    }
}
```

#### visitCode — 메서드 시작 시 바이트코드 삽입

`visitCode` 는 메서드의 바이트코드 시작 지점에서 호출됩니다. 여기서 `timer = timer - System.currentTimeMillis()` 에 해당하는 바이트코드를 삽입합니다.

삽입되는 바이트코드를 단계별로 살펴보겠습니다:

| 단계 | 명령어 | Operand Stack | 설명 |
|------|--------|---------------|------|
| 1 | `GETSTATIC testapp1/AppTest.timer : J` | `[timer]` | 현재 `timer` 값을 Operand Stack 에 push 합니다. |
| 2 | `INVOKESTATIC System.currentTimeMillis()` | `[timer, currentTime]` | 현재 시각을 Operand Stack 에 push 합니다. |
| 3 | `LSUB` | `[timer - currentTime]` | 두 long 값을 빼기합니다 (timer - currentTimeMillis). |
| 4 | `PUTSTATIC testapp1/AppTest.timer : J` | `[]` | 결과를 `timer` 필드에 저장합니다. |

#### visitInsn — 메서드 종료 시 바이트코드 삽입

`visitInsn` 은 피연산자가 없는 명령어를 만날 때마다 호출됩니다. 여기서는 **return 계열 명령어를 만났을 때**만 시간 측정 코드를 삽입합니다.

`opcode >= Opcodes.IRETURN && opcode <= Opcodes.RETURN` 조건은 모든 종류의 return 명령어를 잡기 위한 것입니다. JVM 에는 반환 타입에 따라 `IRETURN`(int), `LRETURN`(long), `FRETURN`(float), `DRETURN`(double), `ARETURN`(참조), `RETURN`(void) 등 여러 return 명령어가 있는데, 이들의 opcode 값이 연속된 범위에 있기 때문에 범위 비교로 한번에 처리할 수 있습니다. `ATHROW`(예외 발생)도 메서드를 빠져나가는 경우이므로 함께 처리합니다.

return 명령어를 만나면 먼저 `timer = timer + System.currentTimeMillis()` 에 해당하는 바이트코드를 삽입합니다:

| 단계 | 명령어 | Operand Stack | 설명 |
|------|--------|---------------|------|
| 1 | `GETSTATIC testapp1/AppTest.timer : J` | `[timer]` | 현재 `timer` 값을 push 합니다. (시작 시 음수가 되어 있음) |
| 2 | `INVOKESTATIC System.currentTimeMillis()` | `[timer, currentTime]` | 현재 시각을 push 합니다. |
| 3 | `LADD` | `[timer + currentTime]` | 두 long 값을 더합니다. 결과가 경과 시간이 됩니다. |
| 4 | `PUTSTATIC testapp1/AppTest.timer : J` | `[]` | 결과를 `timer` 필드에 저장합니다. |

그 다음으로 `System.out.println("elapsed time: " + timer)` 에 해당하는 바이트코드를 삽입합니다. Java 에서 문자열 연결(`+`)은 내부적으로 `StringBuilder` 를 사용하므로, 바이트코드로 직접 `StringBuilder` 를 생성하고 조립합니다:

| 단계 | 명령어 | Operand Stack | 설명 |
|------|--------|---------------|------|
| 5 | `GETSTATIC System.out` | `[PrintStream]` | `System.out` 을 push 합니다. |
| 6 | `NEW StringBuilder` | `[PrintStream, StringBuilder(미초기화)]` | StringBuilder 객체를 생성합니다. |
| 7 | `DUP` | `[PrintStream, SB, SB]` | 생성자 호출과 이후 사용을 위해 참조를 복제합니다. |
| 8 | `LDC "elapsed time: "` | `[PrintStream, SB, SB, "elapsed time: "]` | 문자열 상수를 push 합니다. |
| 9 | `INVOKESPECIAL StringBuilder.<init>(String)` | `[PrintStream, SB]` | StringBuilder 생성자를 호출합니다. |
| 10 | `GETSTATIC AppTest.timer` | `[PrintStream, SB, timer]` | `timer` 값을 push 합니다. |
| 11 | `INVOKEVIRTUAL StringBuilder.append(long)` | `[PrintStream, SB]` | timer 값을 문자열에 추가합니다. |
| 12 | `INVOKEVIRTUAL StringBuilder.toString()` | `[PrintStream, String]` | 완성된 문자열을 얻습니다. |
| 13 | `INVOKEVIRTUAL PrintStream.println(String)` | `[]` | 결과를 출력합니다. |

마지막으로 `mv.visitInsn(opcode)` 를 호출하여 원래의 return 명령어를 실행합니다. 시간 측정 코드는 return **앞에** 삽입되는 것이므로, 원래 메서드의 반환 동작에는 영향을 주지 않습니다.

### JavaAgent

premain 함수는 JVM 에 클래스들이 로드되기 전에 먼저 실행됩니다. 클래스를 변환시킬 때 사용할 ClassTransformer 를 등록합니다.

```java
import java.lang.instrument.Instrumentation;

public class JavaAgent {
    private static Instrumentation instrumentation;

    public static void premain(String options, Instrumentation instrum) {
        JavaAgent.instrumentation = instrum;
        JavaAgent.instrumentation.addTransformer(new ClassTransformer());
    }
}
```

### ClassTransformer — 변환 체인 조립

> **한 줄 요약**: ClassReader → ElapsedTimeCV → AddFieldCV → ClassWriter 순서로 체인을 조립하여, 필드 추가와 메서드 변형을 한 번에 처리합니다.

```java
import java.lang.instrument.ClassFileTransformer;
import java.lang.instrument.IllegalClassFormatException;
import java.security.ProtectionDomain;

import org.objectweb.asm.ClassReader;
import org.objectweb.asm.ClassVisitor;
import org.objectweb.asm.ClassWriter;
import org.objectweb.asm.Opcodes;

public class ClassTransformer implements ClassFileTransformer {
    @Override
    public byte[] transform(ClassLoader loader, String className,
            Class<?> classBeingRedefined, ProtectionDomain protectionDomain,
            byte[] classfileBuffer) throws IllegalClassFormatException {

        if (className.equals("testapp1/AppTest")) {
            ClassReader cr = new ClassReader(classfileBuffer);
            ClassWriter cw = new ClassWriter(cr, ClassWriter.COMPUTE_FRAMES);
            ClassVisitor addFieldCV = new AddFieldCV(cw,
                Opcodes.ACC_STATIC, "timer", "J");
            ClassVisitor elapsedTimeCV = new ElapsedTimeCV(
                Opcodes.ASM5, addFieldCV);
            cr.accept(elapsedTimeCV, 0);
            return cw.toByteArray();
        } else {
            return null;
        }
    }
}
```

변환 체인의 연결 구조를 단계별로 살펴보겠습니다:

1. `ClassWriter cw` — 최종 바이트코드 생성자입니다. `COMPUTE_FRAMES` 옵션을 사용하여 Stack Frame 크기를 자동으로 계산합니다 ([Part 3](/posts/java-bci-asm-part3) 참고).
2. `AddFieldCV(cw, ...)` — ClassWriter 를 감싸서, `static long timer` 필드를 추가합니다.
3. `ElapsedTimeCV(ASM5, addFieldCV)` — AddFieldCV 를 감싸서, `testMethod1` 의 바이트코드를 변형합니다.
4. `cr.accept(elapsedTimeCV, 0)` — ClassReader 가 클래스를 읽으면서 ElapsedTimeCV 에게 이벤트를 보냅니다.

이벤트 흐름 방향은 `ClassReader → ElapsedTimeCV → AddFieldCV → ClassWriter` 입니다. [Part 2](/posts/java-bci-asm-part2)에서 배운 위임 체인이 여러 ClassVisitor 로 확장된 것입니다.

### MANIFEST.MF

```
Manifest-Version: 1.0
Premain-Class: testapp1.agent.JavaAgent
Can-Redefine-Classes: True
```

### 실행

```bash
java -javaagent:/path/to/agent.jar AppMain
```

### 실행 결과

```
App start...
Start Method1
End Method1
elapsed time: 1002
```

원래 소스 코드에 없었던 메서드 실행 시간이 출력되는 것을 확인할 수 있습니다. 소스 코드를 전혀 수정하지 않고, Java Agent 와 ASM 을 통해 바이트코드 수준에서 기능을 추가한 결과입니다.

Eclipse 의 **Bytecode outline** 플러그인을 이용하면 변형 전 클래스의 ASM 코드와 변형 후 클래스의 ASM 코드를 비교해 볼 수 있습니다 (Windows → Show View → Other → Bytecode).

---

## 시리즈 정리

이 시리즈에서는 ASM 을 사용한 Java 바이트코드 조작의 핵심 개념을 다뤘습니다:

- [Part 1](/posts/java-bci-asm-part1) — Java Agent, ClassReader/ClassVisitor/ClassWriter 의 역할
- [Part 2](/posts/java-bci-asm-part2) — 클래스 변형 체인과 위임 구조, 필드 추가
- [Part 3](/posts/java-bci-asm-part3) — JVM 실행 모델 (Stack Frame, Local Variable Array, Operand Stack)
- [Part 4](/posts/java-bci-asm-part4) — MethodVisitor 를 활용한 메서드 변형

바이트코드 조작은 Java Agent, 프로파일러, 모니터링 도구 등 다양한 분야에서 활용됩니다. 이 시리즈가 ASM 라이브러리와 바이트코드 조작의 기초를 이해하는 데 도움이 되었기를 바랍니다. 더 자세한 내용은 [ASM 공식 매뉴얼](https://asm.ow2.io/)을 참고해 주세요.
