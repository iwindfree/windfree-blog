---
title: "[MAUI 기본] 데이터바인딩 - Relative Binding"
author: iwindfree
pubDatetime: 2024-10-15T15:59:47Z
slug: "maui-basic-relative-databinding"
category: "MAUI 기본"
tags: [".net", "maui", "mobile app"]
description: "이번에는 데이터바인딩을 구현하면서 사용하는 Relative Binding 에 대해서 알아보겠습니다. MS 공식 사이트에서는 아래와 같이 Relative Binding 에 대해서 설명되어 있습니다. \".NET Multi-platform App UI .NET MAUI의"
---

이번에는 데이터바인딩을 구현하면서 사용하는 Relative Binding 에 대해서 알아보겠습니다.

MS  공식 사이트에서는 아래와 같이 Relative Binding 에 대해서 설명되어 있습니다.

*".NET Multi-platform App UI (.NET MAUI)의  **상대 바인딩(relative bindings) **은 바인딩 대상의 위치를 기준으로 바인딩 소스를 설정할 수 있는 기능을 제공합니다. 이러한 바인딩은 **RelativeSource 마크업 확장(RelativeSource markup extension) **을 사용하여 생성되며, 바인딩 표현식에서 **Source** 속성으로 설정됩니다."*

개인적으로 Winform 만 사용하다가 WPF, UWP 를 사용하지 않고 바로 XAML 기반의 MAUI 를 접했을 때 이해하기 어려운 부분이라고 생각됩니다. 기존에 많이 사용하셨던 분들에게는 익숙한 개념일 수 있겠습니다.

**Relative Binding**이 생긴 이유는 **복잡한 UI 구조에서 효율적인 데이터 바인딩**을 지원하고, **시각적 트리**나 **템플릿 내부**에서 바인딩 소스를 유연하게 지정할 수 있도록 하기 위해서입니다. 다음은 Relative Binding이 생긴 주요 이유들입니다:

## Relative Binding 필요성

### > 복잡한 UI 구조에서의 바인딩 문제 해결

UI 요소가 계층적으로 중첩되어 있을 때, 특정 요소나 그 상위 요소와 데이터를 바인딩할 필요가 있습니다. 기존의 일반적인 데이터 바인딩은 명시적으로 소스를 지정해야 했으나, 이 경우 특정 요소나 상위 요소에 접근하기가 어렵고 복잡할 수 있었습니다. **Relative Binding**을 사용하면 **현재 컨트롤**이나 **상위 컨트롤**에 쉽게 접근하여 바인딩할 수 있습니다.

### > 템플릿 내부 요소와의 바인딩

템플릿을 사용한 UI에서는 템플릿이 적용된 컨트롤(즉, **템플릿 부모**)에 바인딩하는 것이 매우 일반적입니다. 템플릿 내부에 있는 요소가 템플릿 부모의 속성에 접근해야 하는 경우, 기존의 바인딩 방법으로는 구현하기 어렵습니다. **TemplatedParent** 모드를 사용하면 템플릿 내부 요소가 해당 템플릿이 적용된 부모 요소의 속성에 간편하게 바인딩할 수 있습니다.

### > 상위 요소와의 바인딩

계층적 구조(즉, 시각적 트리)에서 특정 **상위 컨트롤(ancestor)**에 바인딩할 때, 일반적인 바인딩으로는 상위 컨트롤에 쉽게 접근하기 어렵습니다. 이를 해결하기 위해 **FindAncestor** 모드가 도입되었습니다. 이 모드를 사용하면 계층 구조 상위에 위치한 컨트롤을 찾아서 그 컨트롤의 속성이나 **BindingContext**에 쉽게 바인딩할 수 있습니다.

### > 자체 속성 간의 바인딩

UI 컨트롤에서 자신의 한 속성을 다른 속성에 바인딩할 때, 이를 구현하기 위한 방법이 필요합니다. **Self** 모드는 이를 가능하게 해줍니다. 예를 들어, 한 버튼의 IsEnabled 속성을 IsVisible 속성에 바인딩하여 버튼이 보일 때만 활성화되도록 하는 경우가 있습니다. 이런 경우 Relative Binding의 **Self 모드**가 매우 유용합니다.

### > BindingContext 상속 문제 해결

일반적인 데이터 바인딩은 부모에서 자식으로 **BindingContext**가 상속되는데, 때때로 특정 상위 컨트롤의 BindingContext에만 바인딩해야 하는 경우가 있습니다. 이런 경우 **FindAncestorBindingContext** 모드를 사용하면, 특정 상위 컨트롤의 데이터 컨텍스트에만 바인딩할 수 있습니다. 이는 데이터 흐름을 더 정교하게 제어할 수 있도록 도와줍니다.

Relative Binding 의 대표적인 사례를 예제를 통해서 살펴 보도록 하겠습니다.

## **Self Binding 예제**

RelativeSource Self를 사용하는 경우는 매우 드뭅니다. 주로 **현재 컨트롤이나 페이지 자체의 속성**을 **BindingContext**로 바인딩할 때 사용됩니다. 즉, 특정 컨트롤이나 페이지의 속성을 다시 그 자체의 다른 속성에 바인딩하고자 할 때 Self를 사용합니다.

### 페이지나 컨트롤 자체의 속성을 다른 속성에 바인딩

```xml
<ContentPage xmlns="http://schemas.microsoft.com/dotnet/2021/maui"
             xmlns:x="http://schemas.microsoft.com/winfx/2009/xaml"
             x:Class="BindingSample.MainPage"
             Title="{Binding Source={RelativeSource Self}, Path=PageTitle}">
</ContentPage>
```

위 예시에서는 Title 속성을 설정할 때, 같은 페이지에 있는 PageTitle이라는 속성 값을 사용하고 있습니다. 이처럼 **자체 속성 간의 바인딩**을 할 때 RelativeSource Self를 사용합니다.

### 특정 컨트롤의 속성을 해당 컨트롤 자체에 바인딩

```xml
<Button Text="Click Me" IsEnabled="{Binding Source={RelativeSource Self}, Path=IsVisible}" />
```

예를 들어, Button의 IsEnabled 속성을 Button 자체의 다른 속성에 바인딩하고자 할 때 사용할 수 있습니다. 이 경우 Button의 IsEnabled 속성이 IsVisible 속성에 바인딩됩니다. 즉, Button이 보일 때만 클릭 가능하도록 설정하는 예시입니다.

보통은 BindingContext를 ViewModel로 설정하고, 그 속성들에 데이터를 바인딩하는 방식이 더 흔하고 권장됩니다. Self를 사용하는 것보다 **MVVM 패턴**에 따라 ViewModel을 사용하여 데이터를 바인딩하는 방식이 더욱 구조적이고 유지보수에 용이합니다.

## **Bind to an ancestor**

**Bind to an Ancestor**는 UI 요소에서 조상**(ancestor)** 에 해당하는 **상위 요소**에 데이터를 바인딩하는 방식입니다. 즉, **시각적 트리**에서 특정한 부모 요소에 있는 속성이나 데이터에 접근하여 바인딩을 할 때 사용됩니다. 이는 복잡한 계층 구조를 가진 UI에서 자식 요소가 상위 요소의 데이터나 속성에 접근할 필요가 있을 때 유용합니다.  아래를 참고하면 조금 더 이해하기 편할 수 있습니다.

```
ContentPage
└── StackLayout (WidthRequest=200)  <-- 부모 StackLayout
    └── StackLayout                <-- 자식 StackLayout
        └── Label                  <-- 자식 Label
            └── WidthRequest="{Binding Source={RelativeSource AncestorType={x:Type StackLayout}}, Path=WidthRequest}"
```

- ContentPage: 최상위 요소입니다.
- 부모 StackLayout: WidthRequest 속성이 200으로 설정되어 있습니다.
- 자식 StackLayout: 부모 StackLayout 내에 포함된 또 다른 StackLayout입니다.
- Label: 자식 StackLayout 내에 포함된 Label입니다. **이 Label의 WidthRequest 속성은 부모 StackLayout의 WidthRequest 속성에 바인딩되어 있습니다.**

위의 도식화된 이미지는 아래의 소스와 동일합니다.

```xml
<?xml version="1.0" encoding="utf-8" ?>
<ContentPage
    xmlns="http://schemas.microsoft.com/dotnet/2021/maui"
    xmlns:x="http://schemas.microsoft.com/winfx/2009/xaml"
    x:Class="BindingSample.MainPage">

    <StackLayout x:Name="parentLayout" WidthRequest="200">
        <StackLayout>
            <Label Text="Hello, World!"
                   WidthRequest="{Binding Source={RelativeSource AncestorType={x:Type StackLayout}}, Path=WidthRequest}" />
        </StackLayout>
    </StackLayout>
</ContentPage>
```

위 예제는 부모 요소의 속성에 binding 하는 것이고, 부모 요소의 BindingContext 에 binding 을 설정하는 경우도 있습니다.

```xml
<StackLayout>
    <Grid BindingContext="{StaticResource MyViewModel}">
        <Label Text="{Binding Source={RelativeSource AncestorType={x:Type Grid}}, Path=BindingContext.SomeProperty}" />
    </Grid>
</StackLayout>
```

이 경우 Label은 시각적 트리 상의 **Grid** 요소가 사용하는 ViewModel에서 SomeProperty라는 속성에 접근하여 데이터를 바인딩합니다.

한가지 주의할 점은 부모(조상) 요소를 찾기 위해서는 해당 요소의  **Type(예: Grid, StackLayout 등) **을 명시적으로 지정해야 합니다.

위 예제에서는 **AncestorType={x:Type Grid}**  으로 지정한 것을 확인할 수 있습니다. 이를 설정하지 않으면 XAML 구문 분석 과정에서 오류가 발생합니다. 우선 여기까지 해서 Relative Binding 에 대해서 간단히 개념 정리를 해보았습니다.
