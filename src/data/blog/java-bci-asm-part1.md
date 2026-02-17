---
title: "ASM을 이용한 Java 클래스 조작 (Part 1)"
author: iwindfree
pubDatetime: 2019-10-19T09:00:00Z
slug: "java-bci-asm-part1"
category: "JAVA BCI"
series: "JAVA BCI"
seriesOrder: 1
tags: ["java", "bci", "asm"]
description: "BCI(Byte Code Instrumentation)의 개념과 Java Class 구조, Type Descriptor, ASM 라이브러리의 핵심 클래스인 ClassVisitor, ClassReader, ClassWriter를 소개합니다."
canonicalURL: "https://iwindfree.wordpress.com/2019/10/19/manuplating-java-class-with-asm-part1/"
---

**BCI (Byte Code Instrumentation)** 란 Java Class 의 바이트코드를 변조해서, 소스 파일을 수정하지 않고도 특정 기능을 추가하는 기술입니다. 쉽게 말해 소스 코드를 건드리지 않고 프로그램의 동작을 바꾸는 일종의 **"런타임 패치"** 라고 생각하면 됩니다.

BCI 를 가장 적극적으로 활용하는 분야는 APM (Application Performance Management)입니다. 예를 들어 메서드 실행 시간 측정, 예외 발생 추적, SQL 쿼리 캡처 같은 기능을 소스 수정 없이 애플리케이션에 끼워 넣을 수 있습니다. 대표적인 오픈소스 APM인 [scouter](https://github.com/scouter-project) 를 비롯한 다양한 APM 솔루션들이 바로 이 BCI 기술을 기반으로 동작합니다.

이 글에서는 BCI 의 동작 원리를 알아보고, 바이트코드를 다루는 데 널리 쓰이는 라이브러리인 **ASM** 의 핵심 개념을 소개하겠습니다.

## BCI(Byte Code Instrumentation)의 수행 원리

> **한 줄 요약**: Java Agent 를 이용하면, 클래스가 JVM 에 로딩되는 시점에 바이트코드를 가로채서 원하는 대로 변형할 수 있습니다.

먼저 일반적인 Java 프로그램의 실행 흐름을 확인한 뒤, Java Agent 를 사용했을 때 어떤 차이가 생기는지 비교해 보겠습니다. 아래와 같은 간단한 코드가 있다고 가정해 봅시다.

```java
public class MyTest {
    public static void main(String[] args) {
        MyTest test = new MyTest();
        String id  = "id1";
        test.login(id, "P@ssword");
        int money = 100;
        test.doImportTask(id, money);
        System.out.println("Done!!");
    }
}
```

위의 코드를 컴파일하고 실행하면 아래와 같은 순서로 프로그램이 실행됩니다. ClassLoader 가 MyTest 클래스를 JVM 에 로딩하고, main 메서드를 호출하여 "Done!!" 을 출력하는 평범한 흐름입니다.

![java-app1](/images/blog/java-bci/java_class_loading.png)

그런데 JDK 1.5 부터 도입된 **Java Agent** 기능을 활용하면, JVM 에 클래스가 로딩되는 시점에 바이트코드를 가로채서 수정할 수 있습니다. 이번에는 Java Agent 옵션을 추가해서 실행하면 어떤 일이 벌어지는지 살펴보겠습니다.

```bash
java -javaagent:/path/to/agent.jar MyTest
```

`-javaagent` 옵션을 주면 아래와 같은 순서로 실행됩니다:

![javaagent1](/images/blog/java-bci/javaagent.png)

**1단계 — Agent 로딩**: JVM 이 `agent.jar` 를 먼저 로딩합니다. 이 JAR 안의 `MANIFEST.MF` 파일에 `Premain-Class` 항목이 정의되어 있어야 합니다. 예를 들면 다음과 같습니다:

```
Premain-Class: com.example.MyAgent
```

**2단계 — premain 호출**: JVM 이 위에서 지정한 Agent 클래스의 `premain` 메서드를 호출합니다. 이 메서드 안에서 바이트코드를 변조할 **Transformer** 클래스를 등록합니다.

**3단계 — Transformer 적용**: Transformer 가 설정된 이후 JVM 에 로딩되는 모든 클래스는, Transformer 에서 정의한 규칙에 따라 바이트코드가 변조됩니다. 즉, 우리가 원하는 코드가 원래 클래스에 끼워 넣어지는 것입니다.

JDK 1.5 이전에는 ClassLoader 를 직접 수정해야 하는 번거로움이 있었지만, `-javaagent` 옵션 덕분에 클래스 변형 작업이 훨씬 간편해졌습니다. 이제 실제로 Transformer 내부에서 바이트코드를 다룰 때 사용하는 **ASM** 라이브러리에 대해 알아보겠습니다.

**ASM** 은 Java 클래스를 분석하고 변형하는 데 사용하는 오픈소스 프레임워크입니다. ASM 을 이해하려면 먼저 Java 클래스의 내부 구조를 알아야 하므로, 간단히 살펴보겠습니다.

## Java Class

### Class 구조

ASM 은 클래스 파일의 내부 구조를 직접 다루는 도구이기 때문에, 클래스가 내부적으로 어떻게 구성되어 있는지를 먼저 파악해야 합니다. 크게 보면 **클래스 정보**, **필드 목록**, **메서드 목록** 세 부분으로 나눌 수 있습니다:

- **클래스 정보**: 클래스 이름, 부모 클래스(super class), 인터페이스, 접근 제어자(modifier), 어노테이션 등
- **필드(Field)**: 각 필드마다 접근 제어자, 이름, 타입, 어노테이션 정보가 포함됩니다.
- **메서드(Method)**: 각 메서드마다 접근 제어자, 이름, 리턴/파라미터 타입, 어노테이션, 그리고 바이트코드 명령어(instructions)로 표현된 컴파일된 코드가 포함됩니다.

| 항목 | 설명 |
|------|------|
| Modifier, name, super class, interface | 클래스 기본 정보 |
| Constant pool | numeric, string and type constants |
| Source file name | optional |
| Enclosing class reference | outer class 에 대한 참조 |
| Annotation* | 어노테이션 |
| Attribute* | 속성 |
| Inner class* | Inner class 명 |
| Field | Modifier, name, type, annotation*, attribute* |
| Method | Modifiers, name, return and parameter type, annotation*, attribute*, compiled code |

### Type Descriptor

JVM 은 타입을 짧은 기호(Descriptor)로 표현합니다. 처음 보면 낯설지만, 규칙을 알면 금방 익숙해집니다:

- **기본 타입**은 대문자 한 글자로 표현합니다. `I`=int, `F`=float 처럼 대부분 첫 글자를 따옵니다. 다만 `boolean`은 `B`가 이미 `byte`에 사용되었기 때문에 `Z`를 사용하고, `long`도 `L`이 객체 타입 접두사로 사용되기 때문에 `J`를 사용합니다.
- **객체 타입**은 `L패키지/클래스명;` 형태입니다. 예를 들어 `Object`는 `Ljava/lang/Object;` 가 됩니다. 패키지 구분자가 `.` 대신 `/` 인 점에 주의하세요.
- **배열**은 앞에 `[`를 붙입니다. `int[]`는 `[I`, `Object[][]`는 `[[Ljava/lang/Object;` 가 됩니다.

| Type | Descriptor |
|------|-----------|
| `boolean` | `Z` |
| `char` | `C` |
| `byte` | `B` |
| `short` | `S` |
| `int` | `I` |
| `float` | `F` |
| `long` | `J` |
| `double` | `D` |
| `Object` | `Ljava/lang/Object;` |
| `int[]` | `[I` |
| `Object[][]` | `[[Ljava/lang/Object;` |

### Method Descriptor

메서드를 표현할 때는 메서드 이름과 파라미터 이름은 생략되고, **`(파라미터 타입들)리턴타입`** 형식으로 표기합니다.

예를 들어 첫 번째 예시 `(IF)V` 를 풀어보면: 괄호 안의 `I`는 int 파라미터, `F`는 float 파라미터이고, 괄호 밖의 `V`는 void 리턴을 의미합니다. 즉, `void m(int i, float f)` 에 해당합니다.

| Method | Descriptor |
|--------|-----------|
| `void m(int i, float f)` | `(IF)V` |
| `int m(Object o)` | `(Ljava/lang/Object;)I` |
| `int[] m(int i, String s)` | `(ILjava/lang/String;)[I` |
| `Object m(int[] i)` | `([I)Ljava/lang/Object;` |

## ASM Class

ASM 의 핵심은 **Visitor 패턴**입니다. 클래스의 각 구성요소(필드, 메서드, 어노테이션 등)를 하나씩 "방문"하면서 원하는 작업을 수행하는 패턴인데, ASM 에서는 이 패턴을 통해 클래스를 읽고, 분석하고, 변형합니다.

### ClassVisitor

ASM 의 핵심 추상클래스입니다. 이 클래스의 `visitXXX` 메서드들이 클래스의 각 구성요소와 1:1로 대응됩니다. 각 메서드는 단순한 요소(예: 소스 파일명)를 처리할 때는 `void`를 반환하고, 복잡한 요소(예: 메서드, 필드)를 처리할 때는 해당 요소를 더 세밀하게 다룰 수 있는 보조 Visitor(FieldVisitor, MethodVisitor 등)를 반환합니다.

또한 ClassVisitor 는 수신한 모든 메서드 호출을 다른 ClassVisitor 인스턴스에 위임할 수 있어서, 여러 Visitor 를 체인으로 연결하는 이벤트 필터 역할도 합니다.

```java
public abstract class ClassVisitor {
    public ClassVisitor(int api);
    public ClassVisitor(int api, ClassVisitor cv);
    public void visit(int version, int access, String name,
        String signature, String superName, String[] interfaces);
    public void visitSource(String source, String debug);
    public void visitOuterClass(String owner, String name, String desc);
    public AnnotationVisitor visitAnnotation(String desc, boolean visible);
    public void visitAttribute(Attribute attr);
    public void visitInnerClass(String name, String outerName,
        String innerName, int access);
    public FieldVisitor visitField(int access, String name, String desc,
        String signature, Object value);
    public MethodVisitor visitMethod(int access, String name, String desc,
        String signature, String[] exceptions);
    void visitEnd();
}
```

위 코드를 보면 `visitField`, `visitAnnotation`, `visitMethod` 는 각각 `FieldVisitor`, `AnnotationVisitor`, `MethodVisitor` 를 리턴합니다. 왜 그럴까요? 앞서 살펴본 클래스 구조를 떠올려 보면, 메서드 영역에는 바이트코드 명령어까지 포함된 복잡한 구조가 들어있습니다. 이런 복잡한 구성요소를 세밀하게 다루기 위해 별도의 보조 Visitor 를 반환하는 것입니다.

ClassVisitor 의 메서드들은 반드시 아래 순서로 호출되어야 합니다. 이 표기법은 정규 표현식과 비슷한데, `?`는 0~1회, `*`는 0회 이상 호출 가능하다는 뜻입니다:

```
visit visitSource? visitOuterClass? ( visitAnnotation | visitAttribute )*
( visitInnerClass | visitField | visitMethod )*
visitEnd
```

즉, 항상 `visit`으로 시작해서 `visitEnd`로 끝나고, 그 사이에 필드나 메서드 방문이 반복됩니다.

대부분의 메서드는 이름만 봐도 역할을 짐작할 수 있습니다. 이름만으로는 짐작이 어려운 두 메서드만 짚어 두겠습니다:

- **visit**: 클래스의 헤더(버전, 접근 제어자, 이름, 부모 클래스 등)를 방문할 때 가장 먼저 호출됩니다.
- **visitEnd**: 클래스의 모든 구성요소를 방문한 뒤 마지막에 호출됩니다.

### ClassReader

ClassReader 는 컴파일된 클래스 파일(byte array)을 파싱하는 역할을 합니다. 클래스를 읽어가면서 구성요소를 만날 때마다 "필드 발견!", "메서드 발견!" 같은 이벤트를 ClassVisitor 에 전달합니다. 구체적으로는 `accept` 메서드에 ClassVisitor 를 넘겨주면, 파싱 과정에서 해당 Visitor 의 `visitXXX` 메서드들을 자동으로 호출해 줍니다.

ASM 매뉴얼에서는 ClassReader 를 **"이벤트 생산자(event producer)"** 라고 부릅니다.

### ClassWriter

ClassWriter 는 ClassVisitor 의 하위 클래스로, ClassReader 가 보내는 이벤트를 받아서 새로운 바이트코드를 조립하는 역할을 합니다. `toByteArray` 메서드를 호출하면 최종 결과물인 byte array 를 얻을 수 있습니다. ASM 매뉴얼에서는 ClassWriter 를 **"이벤트 소비자(event consumer)"** 라고 부릅니다.

정리하면, ASM 의 전체 파이프라인은 **ClassReader(읽기) → ClassVisitor(분석/변형) → ClassWriter(쓰기)** 흐름으로 동작합니다.

이제 간단한 예제를 통해 ClassReader 와 ClassVisitor 가 실제로 어떻게 동작하는지 확인해 보겠습니다.

## 예제: ClassPrinter

아래 예제는 `java.io.Console` 클래스의 내부 구성요소(필드, 메서드)를 출력하는 간단한 프로그램입니다. ClassVisitor 를 상속받아 직접 구현하면서, 앞서 설명한 개념들이 어떻게 코드로 연결되는지 확인해 보겠습니다.

```java
import java.io.IOException;
import org.objectweb.asm.AnnotationVisitor;
import org.objectweb.asm.Attribute;
import org.objectweb.asm.ClassReader;
import org.objectweb.asm.ClassVisitor;
import org.objectweb.asm.FieldVisitor;
import org.objectweb.asm.MethodVisitor;
import org.objectweb.asm.Opcodes;

public class ClassPrinter extends ClassVisitor {

    public ClassPrinter() {
        super(Opcodes.ASM5);
    }

    public void visit(int version, int access, String name, String signature,
            String superName, String[] interfaces) {
        System.out.println(name + " extends " + superName + " {");
    }

    public void visitSource(String source, String debug) {
    }

    public void visitOuterClass(String owner, String name, String desc) {
    }

    public AnnotationVisitor visitAnnotation(String desc, boolean visible) {
        return null;
    }

    public void visitAttribute(Attribute attr) {
    }

    public void visitInnerClass(String name, String outerName,
            String innerName, int access) {
    }

    public FieldVisitor visitField(int access, String name, String desc,
            String signature, Object value) {
        System.out.println(" " + desc + " " + name);
        return null;
    }

    public MethodVisitor visitMethod(int access, String name, String desc,
            String signature, String[] exceptions) {
        System.out.println(" " + name + desc);
        return null;
    }

    public void visitEnd() {
        System.out.println("}");
    }

    public static void main(String[] args) throws IOException {
        ClassPrinter cp = new ClassPrinter();
        ClassReader cr = new ClassReader("java.io.Console");
        cr.accept(cp, 0);
    }
}
```

코드의 동작을 단계별로 살펴보겠습니다:

1. **ClassPrinter 가 ClassVisitor 를 상속**: `visitField`, `visitMethod` 등 각 구성요소를 만났을 때 실행될 로직을 오버라이드합니다. 여기서는 단순히 이름과 디스크립터를 콘솔에 출력합니다.
2. **ClassReader 생성**: `new ClassReader("java.io.Console")` — 분석할 대상 클래스를 지정합니다.
3. **accept 호출**: `cr.accept(cp, 0)` — ClassReader 가 `java.io.Console` 을 파싱하기 시작하고, 필드를 만나면 `ClassPrinter.visitField()` 를, 메서드를 만나면 `ClassPrinter.visitMethod()` 를 자동으로 호출합니다.

### 실행 결과

```
java/io/Console extends java/lang/Object {
 Ljava/lang/Object; readLock
 Ljava/lang/Object; writeLock
 Ljava/io/Reader; reader
 Ljava/io/Writer; out
 Ljava/io/PrintWriter; pw
 Ljava/util/Formatter; formatter
 Ljava/nio/charset/Charset; cs
 [C rcb
 Z echoOff
 Ljava/io/Console; cons
 Z $assertionsDisabled
 writer()Ljava/io/PrintWriter;
 reader()Ljava/io/Reader;
 format(Ljava/lang/String;[Ljava/lang/Object;)Ljava/io/Console;
 printf(Ljava/lang/String;[Ljava/lang/Object;)Ljava/io/Console;
 readLine(Ljava/lang/String;[Ljava/lang/Object;)Ljava/lang/String;
 readLine()Ljava/lang/String;
 readPassword(Ljava/lang/String;[Ljava/lang/Object;)[C
 readPassword()[C
 flush()V
 encoding()Ljava/lang/String;
 echo(Z)Z
 readline(Z)[C
 grow()[C
 istty()Z
 <init>()V
 <clinit>()V
}
```

결과를 몇 가지만 해석해 보면:

- 첫 줄 `java/io/Console extends java/lang/Object` — Console 이 Object 를 상속한다는 뜻입니다. `visit` 메서드에서 출력된 내용입니다.
- `Ljava/lang/Object; readLock` — Object 타입의 `readLock` 필드입니다. 앞서 배운 Type Descriptor 가 실제로 사용되는 것을 볼 수 있습니다.
- `writer()Ljava/io/PrintWriter;` — 파라미터 없이 `PrintWriter` 를 반환하는 `writer` 메서드입니다. Method Descriptor `()Ljava/io/PrintWriter;` 가 "파라미터 없음 → PrintWriter 반환"을 나타냅니다.
- `[C rcb` — `char[]` 타입의 `rcb` 필드입니다. `[C`는 char 배열의 Type Descriptor 입니다.
