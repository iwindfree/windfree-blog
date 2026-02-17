---
title: "Javassist를 이용한 Java 클래스 조작"
author: iwindfree
pubDatetime: 2019-10-19T13:00:00Z
slug: "java-bci-javassist"
category: "JAVA BCI"
series: "JAVA BCI"
seriesOrder: 5
tags: ["java", "bci", "javassist"]
description: "Javassist 라이브러리와 Java Agent를 사용하여 메서드 수정, 동적 멤버 추가, 시스템 클래스 변조를 구현하는 방법을 설명합니다."
canonicalURL: "https://iwindfree.wordpress.com/2019/10/19/manuplating-java-class-with-javassist/"
---

## BCI (Byte Code Instrumentation)

Java Class 의 bytecode 를 변조해서 소스파일의 수정 없이 특정 기능을 추가하는 작업을 BCI (Byte Code Instrumentation) 이라고 부른다.

BCI 를 대표적으로 활용하는 분야는 APM (Application Performance Management) 분야일 것이다. 대표적인 오픈소스 APM인 [scouter](https://github.com/scouter-project) 를 비롯한 다양한 APM 솔루션들은 BCI 기술을 사용해서 소스 코드의 변경 없이 Application 으로부터 성능 정보를 추출하고 있다.

본 문서는 Java Class 코드를 변형하는 BCI 의 원리와 변형하는데 사용하는 라이브러리 중 Javassist 의 기본적인 활용에 대해서 설명할 것이다.

- Javassist 에 관한 상세 설명은 [공식 홈페이지](http://jboss-javassist.github.io/javassist/) 를 참고하면 된다.
- Java class 를 변형하는 것에 대한 원리와 상세 내용은 본 블로그의 ASM 시리즈를 참고하기 바란다.

## Java Agent 와 Javassist 를 이용한 Class Modify

Javaagent 는 JDK1.5 버전부터 제공되어 지는 기능이다. 간단히 설명하면 Java application 의 main method 전에 실행되는 interceptor 라고 생각하면 된다. 다양한 목적으로 사용될 수 있으며 여기서는 사전에 Java class 를 변조하기 위해 Java Agent 를 사용하는 법에 대해서 간단히 설명하려 한다. BCI (Byte Code Instrumentation) 를 위해서 Javassist 를 사용하였다.

### Java Agent

Java Agent 를 구현하기 위해서는 단순히 아래의 메서드를 구현하면 된다:

```java
public static void premain(String args, Instrumentation inst)
```

그리고 manifest 파일에 아래와 같은 항목을 적어준다:

```
Premain-Class: javaagenttest.MyAgent
```

Premain-Class 에 premain 메서드가 속해있는 class 명을 full name 으로 적어두는 것이다. Javaagent 를 사용하려면 해당 Java Agent 클래스 (위 예에서는 MyAgent) 와 manifest 파일을 jar 파일 형식으로 만들어 배포하여야 하며 실행시에 `-javaagent` 옵션을 명시해 주면 된다.

### ClassFileTransformer

Java Agent 를 사용하여 클래스 파일을 수정하기 위해서는 아래와 같은 절차를 따른다:

1. `ClassFileTransformer` 를 implement 한 class 생성
2. premain 클래스에서 ClassFileTransformer 를 구현한 클래스를 `addTransformer` 메서드를 이용하여 등록
3. ClassFileTransformer 인터페이스의 `transform` 메서드 안에 실제 class 를 변조하는 코드 추가

transform 메서드는 클래스들이 로딩될 때 매번 호출 된다.

```java
public byte[] transform(ClassLoader loader, String className,
        Class<?> classBeingRedefined, ProtectionDomain protectionDomain,
        byte[] classfileBuffer)
```

## 코드 구현

### MyAgent.java

```java
public class MyAgent {
    public static Instrumentation instrumentation;

    public static void premain(String args, Instrumentation inst)
            throws Exception {
        instrumentation = inst;
        instrumentation.addTransformer(new FirstClassTransformer());
    }
}
```

### MANIFEST.MF

```
Manifest-Version: 1.0
Created-By: 1.6.0_33-b03 (Sun Microsystems Inc.)
Premain-Class: javaagent.MyAgent
Can-Redefine-Classes: True
Boot-Class-Path: myagent.jar;javassist.jar
```

### FirstClassTransformer.java

```java
public class FirstClassTransformer implements ClassFileTransformer {

    ClassPool cp;

    public FirstClassTransformer() {
        cp = ClassPool.getDefault();
    }

    public byte[] transform(ClassLoader loader, String className,
            Class<?> classBeingRedefined, ProtectionDomain protectionDomain,
            byte[] classfileBuffer) throws IllegalClassFormatException {

        System.out.println("Class Loader:" + loader.toString()
            + "   Class Name : " + className);

        if (className.equals("javaagent/testclass/Point")) {
            CtClass cc = null;
            try {
                cp.insertClassPath(new LoaderClassPath(loader));
                cc = cp.get("javaagent.testclass.Point");

                //1. method 수정
                CtMethod m = null;
                m = cc.getDeclaredMethod("move");
                m.insertBefore(
                    "{System.out.println($1); System.out.println($2);}");
                m.insertBefore(
                    "{System.out.println(\"I'm modified method in Point class\");}");
                System.out.println("class modified:" + className);

                //2. method 추가
                CtMethod m2 = CtNewMethod.make(
                    "public void hello() " +
                    "{System.out.println(\"hello, I'm point class\");}",
                    cc);
                cc.addMethod(m2);

                //3. field 추가
                CtField cf = new CtField(CtClass.intType, "hiddenValue", cc);
                cf.setModifiers(Modifier.PUBLIC);
                cc.addField(cf);

                return cc.toBytecode();
            } catch (NotFoundException e1) {
                e1.printStackTrace();
            } catch (IOException e) {
                e.printStackTrace();
            } catch (CannotCompileException e) {
                e.printStackTrace();
            }
        }
        return classfileBuffer;
    }
}
```

### Point.java (원본 클래스)

```java
public class Point {
    int x, y;
    public void move(int dx, int dy) {
        x += dx;
        y += dy;
    }
}
```

### 테스트 코드

```java
Point p = new Point();
p.move(3, 3);    // 수정된 메서드

Method[] methodList = Point.class.getMethods();
for (Method m : methodList)   // 추가된 메서드 확인
    System.out.println(m.getName());

System.out.println(
    Point.class.getField("hiddenValue").getName()); // 추가된 필드 확인
```

## System Class 변조

위의 방식을 사용하면 대부분의 클래스를 수정할 수 있지만 시스템 레벨의 클래스는 수정할 수가 없다. 이미 JVM 에 의해서 로딩이 된 상태이기 때문이다.

이미 로딩된 시스템 클래스들을 확인하려면 아래와 같은 코드로 테스트를 해보면 된다:

```java
public static void premain(String args, Instrumentation inst)
        throws Exception {
    instrumentation = inst;

    Class[] classList = instrumentation.getAllLoadedClasses();
    for (Class c : classList) {
        System.out.println("already loaded class : " + c.getName());
    }
}
```

위의 코드를 수행해보면 File I/O, socket 관련 클래스들을 확인해 볼 수 있다.

해당 클래스들을 수정하여 특정 기능을 추가하고 싶은 경우에도 Javassist 를 활용하여 작업할 수 있다. 단, 이러한 경우 `redefineClasses()` 를 사용해야 한다.

```java
public static void premain(String args, Instrumentation inst)
        throws Exception {
    instrumentation = inst;

    // system class 인 경우 vm 에서 로딩 후 다시 재수정
    ClassPool cp = ClassPool.getDefault();
    ArrayList<ClassDefinition> clsDefs = new ArrayList<ClassDefinition>();

    for (Class c : inst.getAllLoadedClasses()) {
        if (c.getName().equals("java.io.FileOutputStream")) {
            System.out.println("Find FileOutputStream Class...");
            CtClass cc = cp.get("java.io.FileOutputStream");
            CtMethod m = cc.getDeclaredMethod("close");
            m.insertBefore(
                "{System.out.println(\"hello\"); " +
                "System.out.println(\"hello2\");}");

            /* System class 인 경우 다시 로딩해서 클래스의
               스키마를 바꾸는 것은 허용되지 않음. */

            // 1. method 추가 실패
            // CtMethod m = CtNewMethod.make(
            //     "public void hello() {System.out.println();}", cc);
            // cc.addMethod(m);

            // 2. field 추가 실패
            // CtField cf = new CtField(
            //     CtClass.intType, "hiddenValue", cc);
            // cf.setModifiers(Modifier.PUBLIC);
            // cc.addField(cf);

            byte[] result = cc.toBytecode();
            if (result != null)
                clsDefs.add(new ClassDefinition(c, result));
        }
    }

    ClassDefinition[] arrClsDefs = clsDefs.toArray(
        new ClassDefinition[clsDefs.size()]);
    instrumentation.redefineClasses(arrClsDefs);
    instrumentation.addTransformer(new FirstClassTransformer());
}
```

### 시스템 클래스 변조 제한사항

System class 인 경우 VM 에서 로딩 후 다시 재수정할 때 클래스의 스키마를 바꾸는 것은 허용되지 않는다. 즉:

- Method 추가 불가
- Field 추가 불가
- 기존 메서드의 동작 수정만 가능
