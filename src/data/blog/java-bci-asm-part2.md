---
title: "ASM을 이용한 Java 클래스 조작 (Part 2)"
author: iwindfree
pubDatetime: 2019-10-19T10:00:00Z
slug: "java-bci-asm-part2"
category: "JAVA BCI"
series: "JAVA BCI"
seriesOrder: 2
tags: ["java", "bci", "asm"]
description: "ClassReader, ClassVisitor, ClassWriter 의 변환 체인 구조를 이해하고, 클래스에 멤버 변수를 추가하는 AddFieldCV 예제를 구현합니다."
canonicalURL: "https://iwindfree.wordpress.com/2019/10/19/manuplating-java-class-with-asm-part2/"
---

## 클래스 변형하기

[Part 1](/posts/java-bci-asm-part1)에서는 ASM 의 핵심 클래스인 ClassReader, ClassVisitor, ClassWriter 의 역할을 살펴보고, 클래스를 읽어서 구성요소를 출력하는 예제까지 다뤘습니다. 이번 Part 2에서는 한 걸음 더 나아가, **클래스를 실제로 변형하는 방법**을 알아보겠습니다.

클래스 변형의 핵심은 다음 세 클래스의 협업 구조에 있습니다:

- **ClassReader** : 이벤트 공급자 — 클래스를 읽으면서 필드, 메서드 등의 구성요소를 만나면 이벤트를 발생시킵니다.
- **ClassVisitor** : 이벤트 필터 — 이벤트를 가로채서 실제 클래스의 내용을 변경하는 역할을 수행합니다.
- **ClassWriter** : 이벤트 소비자 — 전달받은 이벤트를 바탕으로 최종 바이트코드를 생성합니다.

## 클래스 변형 체인

간단한 샘플 코드를 통해서 세 클래스가 어떻게 연결되는지 알아보겠습니다.

### ClassReader → ClassWriter 직접 연결

가장 단순한 형태는 ClassReader 와 ClassWriter 를 직접 연결하는 것입니다. 이 경우 **변형 없이 클래스를 그대로 복사**합니다.



```java
byte[] b1 = ...;
ClassWriter cw = new ClassWriter(0);
ClassReader cr = new ClassReader(b1);
cr.accept(cw, 0);
byte[] b2 = cw.toByteArray();
```

코드를 단계별로 살펴보면:

1. `ClassReader` 가 바이트 배열 `b1` 에서 클래스를 읽습니다.
2. `accept` 메서드에 `ClassWriter` 객체 `cw` 를 넘겨줍니다.
3. ClassReader 가 클래스를 파싱하면서 필드, 메서드 등의 구성요소를 만날 때마다 `cw` 의 `visitXXX` 메서드를 호출합니다. 여기서 중요한 점은, ClassWriter 가 ClassVisitor 를 상속받은 클래스라는 것입니다. 그래서 `accept` 의 파라미터로 넘길 수 있습니다.

이 코드에서 ClassWriter 는 받은 이벤트를 그대로 바이트코드로 조립할 뿐, 중간에 내용을 바꾸는 과정이 없습니다. 결과적으로 `b2` 는 `b1` 과 동일한 클래스가 됩니다.

### ClassReader → ClassVisitor → ClassWriter 연결

클래스를 변형하려면, ClassReader 와 ClassWriter 사이에 **ClassVisitor 를 끼워 넣으면** 됩니다.

![cr-cw-cv](/images/blog/java-bci/cr-cw-cv.jpg)

```java
byte[] b1 = ...;
ClassWriter cw = new ClassWriter(0);
ClassVisitor cv = new MyClassAdaptor(ASM4, cw) { };
ClassReader cr = new ClassReader(b1);
cr.accept(cv, 0);
byte[] b2 = cw.toByteArray();
```

이번에는 `MyClassAdaptor` 라는 ClassVisitor 를 만들어서 중간에 넣었습니다. 여기서 `{ }` 는 Java 의 **익명 클래스(anonymous class)** 문법입니다. 클래스 본체 안에 아무 메서드도 재정의(override)하지 않았기 때문에, ClassReader 가 발생시키는 모든 이벤트가 ClassWriter 에게 그대로 전달됩니다. 즉, 아직은 아무런 변형도 수행하지 않는 상태입니다.

핵심 차이점은 `cr.accept()` 의 인자가 ClassWriter(`cw`)가 아니라 MyClassAdaptor(`cv`)라는 점입니다. ClassReader 가 클래스를 파싱하면서 MyClassAdaptor 의 `visitXXX` 메서드들을 호출하게 됩니다.

이때 동작 방식은 다음과 같습니다:

- MyClassAdaptor 에서 **재정의(override)한** `visitXXX` 메서드가 있으면 → 그 메서드가 실행됩니다.
- **재정의하지 않은** `visitXXX` 메서드는 → 그대로 ClassWriter 에게 위임됩니다.

### 위임 체인 원리

이 위임 구조가 어떻게 만들어지는지 좀 더 자세히 살펴보겠습니다. 마치 **전화 돌리기**와 비슷합니다 — MyClassAdaptor 가 직접 처리할 수 있으면 자기가 받고, 아니면 다음 사람(ClassWriter)에게 넘기는 구조입니다.

단계별로 정리하면 다음과 같습니다:

1. **생성자에서 연결 설정**: `new MyClassAdaptor(ASM4, cw)` 에서 ClassWriter 객체 `cw` 를 넘깁니다. 이 값은 부모 클래스인 ClassVisitor 의 생성자로 전달되어, ClassVisitor 내부의 protected 멤버 변수 `cv` 에 저장됩니다. (ClassWriter 는 ClassVisitor 를 상속받은 클래스이므로, ClassVisitor 타입 변수에 담길 수 있습니다.)
2. **override 된 메서드 호출 시**: MyClassAdaptor 에서 재정의한 `visitXXX` 메서드가 실행됩니다. 이 안에서 원하는 변형을 수행한 뒤, 필요하면 `cv.visitXXX(...)` 를 호출해서 ClassWriter 에게 넘깁니다.
3. **override 되지 않은 메서드 호출 시**: 부모 클래스(ClassVisitor)의 기본 구현이 실행되는데, 기본 구현은 내부 멤버 변수 `cv` (= ClassWriter)에게 호출을 그대로 전달합니다.

이처럼 MyClassAdaptor 가 "중간 다리" 역할을 하면서, 필요한 부분만 가로채고 나머지는 ClassWriter 로 흘려보내는 것이 바로 **위임 체인**의 원리입니다.

### 클래스 버전 변경 예제

> **한 줄 요약**: `visit` 메서드를 재정의해서, 클래스의 Java 버전을 강제로 1.5로 바꾸는 예제입니다.

실제로 클래스를 변형하려면, MyClassAdaptor 에서 수정하고 싶은 항목에 해당하는 `visitXXX` 메서드를 재정의하면 됩니다.

```java
ClassVisitor cv = new MyClassAdaptor(ASM4, cw) {
    @Override
    public void visit(int version, int access, String name,
                      String signature, String superName, String[] interfaces) {
        cv.visit(V1_5, access, name, signature, superName, interfaces);
    }
};
```

`visit` 메서드는 ClassReader 가 클래스를 읽기 시작할 때 가장 먼저 호출되는 메서드로, 클래스의 버전, 접근 제어자, 이름 등 헤더 정보를 전달받습니다. 위 코드에서는 이 메서드를 재정의해서, `cv.visit(V1_5, ...)` 를 호출할 때 첫 번째 파라미터(version)를 원래 값 대신 `V1_5` 로 바꿔 넘기고 있습니다. 나머지 파라미터는 그대로 전달하므로, 결과적으로 **클래스의 버전만 Java 1.5로 변경**됩니다.

> **`cv` 변수 혼동 주의**: 코드에서 `cv.visit(...)` 의 `cv` 는 지역 변수 `ClassVisitor cv = new MyClassAdaptor(ASM4, cw) { ... }` 가 아니라, 익명 클래스 내부에서 접근하는 **부모 클래스 ClassVisitor 의 protected 필드** `this.cv` 입니다. 이 필드에는 생성자에서 넘긴 ClassWriter(`cw`)가 저장되어 있으므로, `cv.visit(...)` 은 결국 **ClassWriter 의 `visit` 을 호출**하는 것입니다. 같은 이름이라 헷갈리기 쉽지만, 바깥의 지역 변수 `cv` 와 안쪽의 필드 `this.cv` 는 서로 다른 것을 가리킵니다.

## 클래스 변형 요약

ClassReader 의 `accept` 메서드가 호출되면 아래와 같은 방식으로 동작합니다.

1. ClassReader 는 주어진 클래스를 읽으면서, 구성요소(field, method 등)를 만날 때마다 파라미터로 넘어온 ClassVisitor(MyClassAdaptor)의 `visitXXX` 메서드를 호출합니다.
2. 해당 메서드가 MyClassAdaptor 에서 **override 되어 있으면** → 재정의한 로직이 실행됩니다. 이 안에서 원하는 변형을 수행한 뒤 ClassWriter 에게 넘길 수 있습니다.
3. 해당 메서드가 **override 되어 있지 않으면** → 부모 클래스(ClassVisitor)의 기본 구현이 실행되어, ClassWriter 에게 그대로 위임됩니다.

핵심을 한 문장으로 정리하면, **ClassReader 와 ClassWriter 사이에 ClassVisitor 를 끼워 넣고, 변형하고 싶은 부분의 `visitXXX` 메서드를 재정의하면 됩니다.**

## 클래스에 멤버 변수 추가해보기

지금까지 클래스 변형의 원리를 알아보았으니, 이제 실전 예제를 작성해 보겠습니다. **기존 클래스에 새로운 멤버 변수를 추가**하는 ClassVisitor 를 구현합니다.

ASM 에서 클래스의 멤버 변수를 만나면 `visitField` 가 호출됩니다. 따라서 새로운 멤버 변수를 추가하려면 `visitField` 를 한 번 더 호출해 주면 됩니다. 다만 이미 같은 이름의 변수가 있으면 중복이 되므로, 다음과 같은 전략을 사용합니다:

1. `visitField` 가 호출될 때마다 추가하려는 변수와 이름이 같은지 확인합니다.
2. 클래스의 모든 구성요소를 다 읽은 뒤 호출되는 `visitEnd` 에서, 중복이 없었다면 그때 `visitField` 를 호출해서 변수를 추가합니다.

이후 예제에서 사용할 `timer` 라는 long 타입의 static 변수를 추가해 보겠습니다.


### AddFieldCV.java

```java
public class AddFieldCV extends ClassVisitor {
    private boolean isFieldAdded = false;
    private String columnName;
    private String desc;
    private int acc;

    public AddFieldCV(ClassVisitor cv, int acc, String col, String desc) {
        super(Opcodes.ASM5, cv);
        this.columnName = col;
        this.acc = acc;
        this.desc = desc;
    }

    @Override
    public FieldVisitor visitField(int access, String name, String desc,
                                    String signature, Object value) {
        if (name.equals(columnName)) {
            isFieldAdded = true;
        }
        return cv.visitField(access, name, desc, signature, value);
    }

    @Override
    public void visitEnd() {
        if (!isFieldAdded) {
            FieldVisitor fv = visitField(this.acc, this.columnName,
                this.desc, null, null);
            if (fv != null) {
                fv.visitEnd();
            }
        }
        cv.visitEnd();
    }
}
```

코드의 동작을 단계별로 살펴보겠습니다:

1. **생성자**: ClassWriter 객체를 `cv` 로 받아 부모 클래스에 전달합니다. 앞서 설명한 위임 체인을 만들기 위한 것입니다. 추가할 변수의 접근 제어자(`acc`), 이름(`columnName`), 타입 디스크립터(`desc`)도 함께 저장합니다. (`columnName` 은 DB 컬럼이 아니라 **추가할 필드의 이름**을 저장하는 변수입니다.)
2. **visitField**: ClassReader 가 기존 필드를 만날 때마다 호출됩니다. 이때 필드 이름이 추가하려는 변수(`columnName`)와 같은지 확인하고, 같으면 `isFieldAdded` 플래그를 `true` 로 설정합니다. 이름과 달리 `isFieldAdded` 는 "필드가 추가되었다"는 뜻이 아니라, **"같은 이름의 필드가 이미 존재한다"** 는 의미의 플래그입니다.
3. **visitEnd**: 클래스의 모든 구성요소를 다 읽은 뒤 호출됩니다. **왜 여기서 추가할까요?** `visitField` 는 ClassReader 가 필드를 만날 때마다 하나씩 호출되므로, 모든 필드를 다 확인하기 전에는 중복 여부를 확신할 수 없습니다. `visitEnd` 는 클래스의 모든 구성요소를 다 읽은 뒤 마지막에 호출되므로, 이 시점이면 중복 검사가 완료된 상태입니다. `isFieldAdded` 가 `false` 라면 (= 같은 이름의 변수가 없었다면) 이제 안전하게 새 필드를 추가합니다.

`visitEnd` 안의 호출 흐름을 좀 더 자세히 살펴보면:

1. `visitField(this.acc, this.columnName, this.desc, null, null)` — 여기서 `visitField` 는 `cv.visitField()` 가 아니라 **자기 자신(`this`)의 `visitField`** 를 호출합니다.
2. 위에서 override 한 `visitField` 가 실행되어, 이번에는 `isFieldAdded` 를 `true` 로 설정하고 `cv.visitField(...)` 를 호출합니다.
3. `cv.visitField(...)` 는 ClassWriter 의 메서드이므로, 이 호출에 의해 **실제로 새 필드가 바이트코드에 기록**됩니다.

이처럼 `this.visitField()` → (override 된) `visitField` → `cv.visitField()` (ClassWriter) 순서로 호출이 이어지면서, 기존의 중복 검사 로직을 재활용하면서도 ClassWriter 에게 정상적으로 위임하는 구조입니다.

### ClassTransformer.java

이제 AddFieldCV 를 실제로 사용하는 Transformer 를 살펴보겠습니다. Part 1에서 설명한 Java Agent 의 Transformer 에 변형 체인을 조립하는 코드입니다.

```java
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
            cr.accept(addFieldCV, 0);
            return cw.toByteArray();
        } else {
            return null;
        }
    }
}
```

변형 체인이 `ClassReader → AddFieldCV → ClassWriter` 순서로 조립되어 있는 것을 확인할 수 있습니다. AddFieldCV 생성자의 마지막 파라미터 `"J"` 는 Part 1에서 다룬 Type Descriptor 로, `long` 타입을 의미합니다. 즉, `testapp1/AppTest` 클래스에 `static long timer` 변수를 추가하는 코드입니다.

좀 더 자세한 내용은 [ASM 공식 매뉴얼](https://asm.ow2.io/)을 참고해 주세요.
