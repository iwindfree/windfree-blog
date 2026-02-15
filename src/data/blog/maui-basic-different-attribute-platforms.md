---
title: "[MAUI 기본] 플랫폼별 다른 속성 지정하기"
author: iwindfree
pubDatetime: 2024-10-15T15:23:00Z
slug: "maui-basic-different-attribute-platforms"
category: "MAUI 기본"
tags: [".net", "maui", "mobile app"]
description: "MAUI 를 사용하여 개발을 진행할 때 플랫폼별로 다른 디자인 속성을 정의해야 할 필요성이 있을 때가 있습니다. 이러한 때 사용할 수 있는 방법을 소개합니다. 두가지 방식을 사용할 수가 있는데요. codebehind 파일에서 프로그래밍적으로 정의할 수도 있고 xaml"
---

MAUI 를 사용하여 개발을 진행할 때 플랫폼별로 다른 디자인 속성을 정의해야 할 필요성이 있을 때가 있습니다. 이러한 때 사용할 수 있는 방법을 소개합니다.  두가지 방식을 사용할 수가 있는데요. codebehind 파일에서 프로그래밍적으로 정의할  수도 있고 xaml 파일에서 정의할 수도 있습니다.

(MainPage.xaml)

```xml
<?xml version="1.0" encoding="utf-8" ?>
<ContentPage
    x:Class="MyFirstMauiApp.MainPage"
    xmlns="http://schemas.microsoft.com/dotnet/2021/maui"
    xmlns:x="http://schemas.microsoft.com/winfx/2009/xaml">   
    <ScrollView>
        <VerticalStackLayout
            x:Name="VStackLayout"
            Padding="30,0"
            Spacing="25">
            <Image
                Aspect="AspectFit"
                HeightRequest="185"
                SemanticProperties.Description="dot net bot in a race car number eight"
                Source="dotnet_bot.png" />

            <Label
                SemanticProperties.HeadingLevel="Level1"
                Style="{StaticResource Headline}"
                Text="Hello, World!" />

            <Label
                SemanticProperties.Description="Welcome to dot net Multi platform App U I"
                SemanticProperties.HeadingLevel="Level2"
                Style="{StaticResource SubHeadline}"
                Text="Welcome to 
.NET Multi-platform App UI" />

            <Button
                x:Name="CounterBtn"
                Clicked="OnCounterClicked"
                HorizontalOptions="Fill"
                SemanticProperties.Hint="Counts the number of times you click"
                Text="Click me" />
        </VerticalStackLayout>
    </ScrollView>
</ContentPage>
```

기본 템플릿에서 제공되는 MainPage.xaml 입니다. MainPage.xmal.cs 파일에 플랫폼별로 VerticalStackLayout 의 배경색을 변경하는 코드를 추가해보겠습니다. codebehind 소스 파일에서 참조할 수 있도록 x:Name 을 이용하여 VerticalStackLayout 의 이름을 "VStackLayout" 으로 지정하였습니다.

(MainPage.xaml.cs)

```csharp
 public MainPage()
 {
     InitializeComponent();
     VStackLayout.BackgroundColor = DeviceInfo.Platform == DevicePlatform.Android ? Colors.LightGoldenrodYellow : Colors.LightSkyBlue;
 }
```

실행해서 결과를 확인해보면 안드로이드 에뮬레이터에서는 배경색이 변경된 것을 확인할 수 있습니다.

![](/images/blog/2024/10/yellow-car-1.png)

이번에는 직접 xaml 파일에서 변경해보도록 하겠습니다.

```xml
<?xml version="1.0" encoding="utf-8" ?>
<ContentPage
    x:Class="MyFirstMauiApp.MainPage"
    xmlns="http://schemas.microsoft.com/dotnet/2021/maui"
    xmlns:x="http://schemas.microsoft.com/winfx/2009/xaml">

    <ScrollView>
        <VerticalStackLayout
            x:Name="VStackLayout"
            Padding="30,0"
            Spacing="25" 
            BackgroundColor="{OnPlatform Android=Green, iOS=Blue}">
            <Image
                Aspect="AspectFit"
                HeightRequest="185"
                SemanticProperties.Description="dot net bot in a race car number eight"
                Source="dotnet_bot.png" />

            <Label
                SemanticProperties.HeadingLevel="Level1"
                Style="{StaticResource Headline}"
                Text="Hello, World!" />

            <Label
                SemanticProperties.Description="Welcome to dot net Multi platform App U I"
                SemanticProperties.HeadingLevel="Level2"
                Style="{StaticResource SubHeadline}"
                Text="Welcome to 
.NET Multi-platform App UI" />

            <Button
                x:Name="CounterBtn"
                Clicked="OnCounterClicked"
                HorizontalOptions="Fill"
                SemanticProperties.Hint="Counts the number of times you click"
                Text="Click me" />
        </VerticalStackLayout>
    </ScrollView>
</ContentPage>
```

```
BackgroundColor="{OnPlatform Android=Green, iOS=Blue}"
```

BackgroundColor 속성을 추가해서 xaml 파일을 사용해서 플랫폼별로 배경색을 조절하는 것을 확인할 수 있습니다. 실제 실행 결과는 아래와 같습니다.

![](/images/blog/2024/10/green-car-1.png)
