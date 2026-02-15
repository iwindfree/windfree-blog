---
title: "[MAUI 기본] Mac 기반 Window VM 에서 Android Emulator 사용하기"
author: iwindfree
pubDatetime: 2024-10-15T15:09:30Z
slug: "maui-basic-using-emulator-with-parallels"
category: "MAUI 기본"
tags: [".net", "maui", "mobile app"]
description: "Mac 에서 일반적으로 Window 를 사용하기 위해서는 일반적으로 Parallels 를 사용하게 된다. .NET MAUI 를 공부하면서 Android 에뮬레이터를 사용해야 했지만, Parallel 을 사용하는 경우 ARM 기반의 Windows 환경에서는"
---

Mac 에서 일반적으로 Window 를 사용하기 위해서는 일반적으로 Parallels 를 사용하게 된다. .NET MAUI 를 공부하면서 Android 에뮬레이터를 사용해야 했지만, Parallel 을 사용하는 경우 ARM 기반의 Windows 환경에서는 Android Emulator 를 실행하는 것이 쉽지가 않다.  이러한 경우, VM 기반으로 실행하는 Windows 에서 Mac 에서 실행되는 Android Emulator 에 연결하는 방식으로 Android Emulator 를 사용할 수 있는 방법이 있어 정리해 두었다.

## On Mac

### Androd SDK 설치

Mac 에서 Android SDK 를 설치하는 방식은 다양하다. 아래 방법 중 편한 방식을 선택하면 된다.

- Android Studio 설치
- IntelliJ 설치 후 Android project 구성
- .NET 및 .NET MAUI workloads 설치 ([https://learn.microsoft.com/en-us/dotnet/maui/get-started/installation?view=net-maui-8.0&tabs=visual-studio-code)](https://learn.microsoft.com/en-us/dotnet/maui/get-started/installation?view=net-maui-8.0&tabs=visual-studio-code)
- 직접 Android SDK 설치

AVD (Android Virtual Device) 를 생성하거나 관리를 편하게 하기 위해서 일반적으로 IDE 를 설치하면서 Android SDK 를 동시에 설치하는 방법이 편리하다.  Android SDK 의 설치 경로는 일반적으로 아래와 같다. [](https://learn.microsoft.com/en-us/dotnet/maui/get-started/installation?view=net-maui-8.0&tabs=visual-studio-code)

```bash
/Users/<username>/Library/Android/sdk
```

### Android Emulator 실행

사용하는 IDE 를 이용하여 Android Emulator 를 실행한다.  사용하는 IDE 별로 방법은 상이할 수 있다. 여기서는 android studio 를 사용하여 Android emulator 를 실행할 것이다.

- Android studio 실행
- 프로젝트 생성 (실제 프로젝트를 사용할 것은 아니기 때문에 단순히 No-Activity 로 생성해도 된다.)
- Tools -> Device Manager 실행
- 등록되어 있는 virtual device 실행

![](/images/blog/2024/10/android-device.png)

### ADB Server 종료

Mac 에서 현재 실행되고 있는 adb server 를 종료한다.

```
cd /Users/<username>/Library/Android/sdk/platform-tools
adb kill-server
```

Mac에서 실행 중인 Android 에뮬레이터에 Windows 가상 머신(VM)에서 연결할 때, adb kill-server 명령을 사용하는 이유는 ADB(Android Debug Bridge) 서버를 재시작하여 혹시나 발생할 수 있는 연결 문제를 해결하기 위함이다. Mac과 Windows VM 사이에서 네트워크나 포트 포워딩 문제로 인해 ADB가 제대로 동작하지 않을 수 있다. 이때 adb kill-server를 실행해 현재 ADB 서버를 종료한 후, 새로운 연결을 설정할 때 ADB 서버가 자동으로 다시 시작되게 할 수 있다. 이를 통해 에뮬레이터와 VM 간의 원활한 통신이 가능하게 된다.

현재 동작하고 있는 Android emulator 를 확인하려면 아래 명령을 사용한다.

```
./adb devices
```

아래와 비슷한 결과를 확인할 수 있다.

```
List of devices attached
emulator-5554	device
```

아래 명령을 이용해서 Android Emulator 의 동작 여부 및 포트를 확인할 수 있다.

```
lsof -iTCP -sTCP:LISTEN -P | grep 'emulator\|qemu'

emulator6 94105 macuser   20u  IPv4 0xa8dacfb1d4a1b51f      0t0  TCP localhost:5555 (LISTEN)
emulator6 94105 macuser   21u  IPv4 0xa8dacfb1d845a51f      0t0  TCP localhost:5554 (LISTEN)
```

**ADB와 콘솔 포트**: 포트 **5554**는 Android 에뮬레이터의 콘솔에 연결되기 위한 포트이고, **5555**는 ADB(Android Debug Bridge)를 통해 에뮬레이터에 접근하는 포트이다. 두 포트는 항상 짝을 이루며, 에뮬레이터가 실행될 때마다 생성된다.

### MAC 에서 원격 로그인 활성화

VM  기반 윈도우에서 MAC 으로 ssh 연결을 허용하기 위해서 '원격 로그인' 기능을 활성화 한다.

![](/images/blog/2024/10/enable-mac-remote-login.png)

Mac 에서 해야 할 일은 이것이 전부다. 이제 Windows 에서 Android Emulator 에 접속하기 위한 작업을 수행한다.

## On Windows

### 터널링 설정하기

```
ssh -L localhost:15555:127.0.0.1:5555 mac-user@ip-address-of-the-mac
```

ssh -L localhost:15555:127.0.0.1:5555 mac-username@ip-address-of-the-mac는 Windows VM에서 Mac에 있는 Android 에뮬레이터에 연결하기 위해 사용하는 명령어다. 이 명령어는 로컬 포트 포워딩을 설정하는데, 각 부분의 설명은 다음과 같다.

1. **ssh -L** 는 로컬 포트 포워딩을 설정하는 옵션이다. 이를 통해 로컬 시스템에서 특정 포트로 들어오는 트래픽을 원격 시스템의 다른 포트로 전달한다.
2. **localhost:15555** 는 로컬 포트를 나타낸다. 여기서는 Windows VM에서 로컬 포트 15555를 사용해 Mac으로 트래픽을 전달한다.
3. **127.0.0.1:5555** 는 원격 Mac의 포트다. 여기서 127.0.0.1은 Mac의 루프백 주소이고, 5555는 Android 에뮬레이터의 ADB 포트다.
4. mac-user@ip-address-of-the-mac는 SSH로 Mac에 접속하기 위한 사용자 이름과 IP 주소를 나타낸다.

이 명령어를 통해 Windows VM의 포트 15555가 Mac의 5555 포트로 연결되어 두 시스템 간에 에뮬레이터를 통해 통신할 수 있게 된다.

### ADB Connect

터널링이 실행되었으면, Windows 에 설치된 adb 명령어를 사용하여 MAC 에 있는 Android Emulator 에 접속한다. 참고로 Windows 환경에서 기본으로 설치되는 android sdk 경로는 "C:\Program Files (x86)\Android\android-sdk" 이며, adb 는 해당 경로 하위의 platform-tools 폴더에 위치한다.

```
adb connect localhost:15555
```

터널링을 통하여 설정한 로컬포트 15555 로 접속하면 MAC 에서 실행되고 있는 Android Emulator 에 접속할 수 있다.

### 모바일 앱 실행하여 확인하기

Visual Studiio 에서 adb connect 이 후 실행가능 타겟 플랫폼을 보면 "Android 로컬 디바이스" 에 MAC 에서 실행되고 있는 Android Emulator 를 확인할 수 있다.

![](/images/blog/2024/10/vstudio-adv.png)

이제 MAUI 앱을 실행하면 MAC 의 안드로이드 에뮬레이터에서 해당 앱이 실행되는 것을 확인할 수 있다.
