---
title: "[MAUI 기본] 공유리소스 활용하기"
author: iwindfree
pubDatetime: 2024-10-15T15:27:48Z
slug: "maui-basic-using-shared-resources"
category: "MAUI 기본"
tags: [".net", "maui", "mobile app"]
description: "어플리케이션을 개발할 때 소스의 여러곳에서 공통적으로 사용하는 리소스를 정의하여 사용하는 경우가 많습니다. 이번에는 공유리소스 파일을 생성하고, 리소스파일에 정의한 내용을 xaml 에서 사용하는 간단한 예제를 설명하겠습니다. SharedResources.cs"
---

어플리케이션을 개발할 때 소스의 여러곳에서 공통적으로 사용하는 리소스를 정의하여 사용하는 경우가 많습니다. 이번에는 공유리소스 파일을 생성하고, 리소스파일에 정의한 내용을 xaml 에서 사용하는 간단한 예제를 설명하겠습니다.

[SharedResources.cs]

```csharp
namespace MyFirstMauiApp.Resources
{
    static class SharedResources
    {
        public static readonly Color FontColor = Color.FromRgb(0, 0, 255);
    }
}
```

간단하게 static class 를 생성하고 readonly 로 FontColor  를 선언하였습니다. 이제 xaml 파일에서 해당 속성을 참조하여 텍스트 색상을 변경하는 예제를 살펴 보겠습니다.

```xml
<?xml version="1.0" encoding="utf-8" ?>
<ContentPage
    x:Class="MyFirstMauiApp.MainPage"
    xmlns="http://schemas.microsoft.com/dotnet/2021/maui"
    xmlns:x="http://schemas.microsoft.com/winfx/2009/xaml"
    xmlns:resources="clr-namespace:MyFirstMauiApp.Resources">

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
                Text="Hello, World!"
                TextColor="{x:Static Member=resources:SharedResources.FontColor}" />

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

위의 소스를 보면 ContentPage 의 namespace 선언 부분에 아래의 내용이 추가되었습니다.

```
    xmlns:resources="clr-namespace:MyFirstMauiApp.Resources"
```

이 부분은 사용자 정의 네임스페이스를 지정하여, MyFirstMauiApp.Resources 네임스페이스의 리소스를 사용할 수 있게 합니다.

그리고, Label 선언부에서 TextColor 를 리소스파일을 이용해서 정의한 것을 확인할 수 있습니다.

```
 <Label
     SemanticProperties.HeadingLevel="Level1"
     Style="{StaticResource Headline}"
     Text="Hello, World!"
     TextColor="{x:Static Member=resources:SharedResources.FontColor}" />
```

실행 결과를 확인해보면 해당 부분의 색상이 변경된 것을 확인할 수 있습니다.

![](/images/blog/2024/10/yellow-car-1.png)
