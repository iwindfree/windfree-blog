---
title: "[MAUI 기본] 데이터바인딩 - Compiled Binding"
author: iwindfree
pubDatetime: 2024-10-15T16:05:49Z
slug: "maui-basic-compiled-databinding"
category: "MAUI 기본"
tags: [".net", "maui", "mobile app"]
description: "이번에는 Compiled Binding 에 대해서 알아보도록 하겠습니다. Compiled Binding의 주요 특징 1. 컴파일 타임 유효성 검사: 일반적인 데이터 바인딩은 런타임에서 바인딩 오류를 발견하지만, Compiled Binding은 컴파일 타임에 바인딩"
---

이번에는 Compiled Binding 에 대해서 알아보도록 하겠습니다.

## Compiled Binding의 주요 특징

1. **컴파일 타임 유효성 검사**:

일반적인 데이터 바인딩은 런타임에서 바인딩 오류를 발견하지만, **Compiled Binding**은 컴파일 타임에 바인딩 오류를 감지합니다. 따라서 코드 작성 시점에 잘못된 바인딩 표현식이 있으면 즉시 알림을 받아 수정할 수 있습니다.
2. 예를 들어, XAML에서 잘못된 속성이나 바인딩 경로를 사용하면 **빌드 오류**로 즉시 확인할 수 있어, 디버깅 시간이 단축됩니다.

## Compiled Binding 사용법

.NET MAUI에서 **Compiled Binding**을 사용하려면, XAML에서** x:DataType 또는 x:Type** 마크업 확장을 명시적으로 설정해야 합니다. 이 속성은 XAML 바인딩에서 사용할 데이터의 타입을 컴파일러에 알려줍니다. 컴파일된 바인딩 표현식은 **소스 속성에서 값을 가져와** 마크업에서 지정한 **타겟 속성에 값을 설정하는** 컴파일된 코드를 생성합니다. 또한, 바인딩 표현식에 따라 **소스 속성의 값이 변경되면 이를 관찰하여 타겟 속성을 갱신**하거나, **타겟에서 소스로 값이 전달되는 양방향 바인딩**도 지원할 수 있습니다.  간단한 소스로 확인해 보겠습니다.

```xml
<ContentPage xmlns="http://schemas.microsoft.com/dotnet/2021/maui"
             xmlns:x="http://schemas.microsoft.com/winfx/2009/xaml"
             xmlns:local="clr-namespace:MyApp.ViewModels"
             x:Class="MyApp.MainPage"
             x:DataType="local:MainViewModel">
    
    <StackLayout>
        <!-- Compiled Binding Example -->
        <Label Text="{Binding Name}" />
        <Label Text="{Binding Age}" />
    </StackLayout>
    
</ContentPage>
```

- x:DataType="local:MainViewModel"을 통해, MainViewModel을 바인딩할 데이터 타입으로 지정합니다.
- 컴파일러는 Name과 Age가 MainViewModel의 속성인지 컴파일 타임에 확인하고, 잘못된 속성이 있으면 빌드 오류를 발생시킵니다.

## x:DataType 재정의

**x:DataType 속성은 뷰 계층 구조의 어떤 지점에서도 재정의할 수 있습니다. **즉, 상위 요소에서 설정된 x:DataType을 하위 요소에서 변경하여, 특정 하위 요소에 대해 다른 데이터 타입에 바인딩하도록 지정할 수 있습니다. 이렇게 함으로써, 다양한 데이터 모델을 사용하는 복잡한 UI 구조에서도 유연하게 바인딩을 관리할 수 있습니다. 이 기능은 특히 MVVM 패턴을 따르는 애플리케이션에서 유용하며, 각 요소가 필요한 데이터 타입에 맞춰 바인딩을 설정할 수 있도록 도와줍니다. 간단한 예를 통해서 확인해 보겠습니다. 이전 강의에서 사용했던 소스의 일부입니다.

```xml
<ContentPage
    x:Class="BindingSample.MainPage"
    xmlns="http://schemas.microsoft.com/dotnet/2021/maui"
    xmlns:x="http://schemas.microsoft.com/winfx/2009/xaml"
    xmlns:viewmodel="clr-namespace:BindingSample.ViewModels"
    x:DataType="viewmodel:PeopleViewModel"
    Title="{Binding Title}">
    <ContentPage.BindingContext>
        <viewmodel:PeopleViewModel />
    </ContentPage.BindingContext>
    <StackLayout>
        <CollectionView ItemsSource="{Binding Peoples}" SelectionMode="Single">
            <CollectionView.ItemTemplate>
                <DataTemplate>
                    <StackLayout Padding="10" Spacing="5">
                        <Label Text="{Binding Name}" />
                        <Label Text="{Binding Age}" />
                    </StackLayout>
                </DataTemplate>
            </CollectionView.ItemTemplate>
        </CollectionView>
    </StackLayout>
</ContentPage>
```

전체 소스에 대한 설명은 이전 포스팅중에   [[MAUI 기본] 데이터바인딩 – 기본 개념](http://iwindfree.mycafe24.com/maui-%ea%b8%b0%eb%b3%b8-%eb%8d%b0%ec%9d%b4%ed%84%b0%eb%b0%94%ec%9d%b8%eb%94%a9-%ea%b8%b0%eb%b3%b8-%ea%b0%9c%eb%85%90/) 을 참고하시면 됩니다.

이전 소스에 정확히 한줄만 더 추가하였습니다.

```
x:DataType="viewmodel:PeopleViewModel"
```

위에서 설명한 바와 같이 x:DataType 을 명시함으로써, Compiled Binding 을 지원하게 되었습니다. 이제 한번 해당 소스를 빌드를 해보면  아래와 같은 오류가 발생되며 빌드가 실패하게 됩니다.

```
Binding: Property "Name" not found on "BindingSample.ViewModels.PeopleViewModel".
```

왜 이런 오류가 갑자기 생겼을까요? 우리가 추가한 것은 x:DataType 을 이용한 것밖에 없는데요. 위 예제에서 DataTemplate은  부모 스코프에서 잘못된 x:DataType을 상속받았습니다. 결국 위에 선언한 PeopleViewModel 을 데이터타입으로 인식해서 Name 과 Age 속성을 PeopleViewModel 에서 찾으려고 시도하다가 에러가 발생한 것입니다.  이렇게 되면, 데이터 객체에 대한 올바른 타입 정보가 부족해 바인딩 오류가 발생하게 됩니다.  그러면 어떻게 이런 오류를 해결해야 할까요?

DataTemplate에서 컴파일된 바인딩을 사용할 때는 템플릿의 데이터 객체의 타입을 x:DataType 속성으로 선언해야 합니다. 즉, 아래와 같이 코드를 변경하면 됩니다.

```xml
 <DataTemplate x:DataType="model:Employee">
     <StackLayout Padding="10" Spacing="5">
         <Label Text="{Binding Name}" />
         <Label Text="{Binding Age}" />
     </StackLayout>
 </DataTemplate>
```

DataTemple 태그안에 x:DataType 을 선언한 것을 확인할 수 있습니다. model 을 네임스페이스 접두어로 사용하기 위해서 아래 코드도 추가되어야 합니다.

```
xmlns:model="clr-namespace:BindingSample.Models"
```

수정된 전체 코드는 아래와 같습니다. 크게 변한 것은 없으니 간단히 살펴보시면 될 것 같습니다 .

```xml
<?xml version="1.0" encoding="utf-8" ?>
<ContentPage
    x:Class="BindingSample.MainPage"
    xmlns="http://schemas.microsoft.com/dotnet/2021/maui"
    xmlns:x="http://schemas.microsoft.com/winfx/2009/xaml"
    xmlns:model="clr-namespace:BindingSample.Models"
    xmlns:viewmodel="clr-namespace:BindingSample.ViewModels"
    Title="{Binding Title}"
    x:DataType="viewmodel:PeopleViewModel">
    <ContentPage.BindingContext>
        <viewmodel:PeopleViewModel />
    </ContentPage.BindingContext>
    <StackLayout>
        <CollectionView ItemsSource="{Binding Peoples}" SelectionMode="Single">
            <CollectionView.ItemTemplate>
                <DataTemplate x:DataType="model:Employee">
                    <StackLayout Padding="10" Spacing="5">
                        <Label Text="{Binding Name}" />
                        <Label Text="{Binding Age}" />
                    </StackLayout>
                </DataTemplate>
            </CollectionView.ItemTemplate>
        </CollectionView>
    </StackLayout>
</ContentPage>
```

## 주의할 점

MS 공식 사이트에서는 CompiledBinding 의 제약 사항에 대해서 아래와 같이 설명하고 있습니다.

*"컴파일된 바인딩은 Source 속성을 정의하는 XAML 바인딩 표현식에 대해 비활성화됩니다. 그 이유는 Source 속성이 항상 x:Reference 마크업 확장을 사용하여 설정되기 때문이며, 이 속성은 컴파일 타임에 해결할 수 없습니다. 따라서, Source 속성을 사용하는 바인딩은 런타임에 해석되므로 컴파일된 바인딩을 지원하지 않습니다.*

*또한, XAML에서 컴파일된 바인딩은 현재 멀티 바인딩(multi-bindings)에서도 지원되지 않습니다. 멀티 바인딩은 여러 소스 속성을 단일 타겟 속성에 바인딩하는 방식으로, 이 또한 컴파일 타임에 해결할 수 없는 특성 때문에 컴파일된 바인딩을 사용할 수 없습니다."*

이러한 고려사항을 정확히 이해해서  데이터바인딩을 설계하시기 바랍니다.
