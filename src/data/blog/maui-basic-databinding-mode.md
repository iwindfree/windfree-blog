---
title: "[MAUI 기본] 데이터바인딩 - 바인딩 모드"
author: iwindfree
pubDatetime: 2024-10-15T15:56:42Z
slug: "maui-basic-databinding-mode"
category: "MAUI 기본"
tags: [".net", "maui", "mobile app"]
description: "이번에는 데이터 바인딩 모드에 대해서 간단하게 정리해 보겠습니다. 데이터 바인딩 모드는 데이터가 View뷰, UI와 ViewModel데이터 소스 사이에서 어떻게 흐를지를 결정하는 방식입니다. 여기에는 다양한 바인딩 모드가 있으며, 각 모드는 데이터 흐름의 방향을"
---

이번에는 데이터 바인딩 모드에 대해서 간단하게 정리해 보겠습니다.

데이터 바인딩 모드는 **데이터가 View(뷰, UI)와 ViewModel(데이터 소스) 사이에서 어떻게 흐를지를 결정하는 방식**입니다. 여기에는 다양한 바인딩 모드가 있으며, 각 모드는 데이터 흐름의 방향을 정의합니다. 자주 사용되는 바인딩 모드는 다음과 같습니다.

![](/images/blog/2024/10/databinding-mode.png)

### 1. **OneWay** (단방향)

- **데이터 흐름**: ViewModel → View
- **설명**: 데이터가 **ViewModel(소스)**에서 **View(UI)**로만 전달됩니다. UI 요소는 데이터 소스를 관찰하여 변경 사항을 반영하지만, 반대로 UI에서 변경된 값이 ViewModel로 전달되지는 않습니다.
- **사용 예시**: Label에 데이터를 표시할 때.

```
<Label Text="{Binding UserName, Mode=OneWay}" />
```

### 2. **TwoWay** (양방향)

- **데이터 흐름**: ViewModel ↔ View
- **설명**: 데이터가 **양방향**으로 흐릅니다. ViewModel의 변경 사항이 View에 반영되고, 사용자가 UI에서 값을 수정하면 그 값이 다시 ViewModel로 전달됩니다.
- **사용 예시**: 사용자가 직접 값을 입력하거나 수정할 수 있는 UI 요소에서 사용 (예: Entry, Slider 등).

```
<Entry Text="{Binding UserName, Mode=TwoWay}" />
```

### 3. **OneWayToSource** (단방향 소스로)

- **데이터 흐름**: View → ViewModel
- **설명**: 데이터가 **View(UI) **에서 **ViewModel(소스) **로만 전달됩니다. 주로 UI에서 발생한 변경 사항을 ViewModel에 전달하고 싶을 때 사용됩니다. 반대로 ViewModel의 값이 UI에 반영되지는 않습니다.
- **사용 예시**: ListView에서 선택된 항목을 ViewModel에 전달할 때 사용.

```
<ListView SelectedItem="{Binding SelectedItem, Mode=OneWayToSource}" />
```

### 4. **OneTime** (한 번만)

- **데이터 흐름**: ViewModel → View (한 번만)
- **설명**: 바인딩이 설정된 후, 데이터가 **초기**에만 ViewModel에서 View로 전달됩니다. 이후 ViewModel에서 값이 변경되더라도 UI에는 반영되지 않습니다. 주로 변경되지 않을 초기 데이터 로딩 시 사용됩니다.
- **사용 예시**: 초기화 후 값이 변하지 않는 경우 (예: 초기 화면 로드 시 한 번만 데이터를 표시).

```
<Label Text="{Binding UserName, Mode=OneTime}" />
```

### 5. **Default** (기본 모드)

- **데이터 흐름**: 각 속성에 따라 다름
- **설명**: 각 속성마다 **기본 바인딩 모드**가 정의되어 있으며, 이를 자동으로 따릅니다. 대부분의 읽기 전용 속성은 OneWay, 쓰기 가능한 속성은 TwoWay로 설정됩니다.

### 요약

- **OneWay**: ViewModel → View (단방향, 주로 읽기 전용)
- **TwoWay**: ViewModel ↔ View (양방향, 읽고 수정 가능)
- **OneWayToSource**: View → ViewModel (단방향, UI 이벤트에 따라 소스 업데이트)
- **OneTime**: ViewModel → View (초기 값만 한 번 설정, 이후 변화 없음)
- **Default**: 속성의 기본 설정에 따라 달라짐

이렇게 각 바인딩 모드는 데이터가 흐르는 방향과 방식을 설정하는 중요한 개념이며, 상황에 맞는 바인딩 모드를 선택하는 것이 중요합니다.
