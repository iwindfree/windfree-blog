---
title: "[MAUI 기본] Page Navigation"
author: iwindfree
pubDatetime: 2024-10-15T15:39:42Z
slug: "maui-basic-page-navigation"
category: "MAUI 기본"
tags: [".net", "maui", "mobile app"]
description: "일반적으로 어플리케이션은 여러개의 페이지로 구성되어 있습니다. 이번 장에서는 여러개의 페이지를 구성하는 방법에 대해서 간단히 살펴보겠습니다. 우선 샘플 소스를 보기 전에 MAUI 프로젝트를 생성하면 기본적으로 만들어지는 AppShell 에 대해서 알아보도록"
---

일반적으로 어플리케이션은 여러개의 페이지로 구성되어 있습니다. 이번 장에서는 여러개의 페이지를 구성하는 방법에 대해서 간단히 살펴보겠습니다. 우선 샘플 소스를 보기 전에 MAUI 프로젝트를 생성하면 기본적으로 만들어지는 AppShell 에 대해서 알아보도록 하겠습니다.

**AppShell은 .NET MAUI 애플리케이션에서 네비게이션 구조를 정의하고 관리하는 데 사용되는 클래스입니다**. AppShell을 사용하면 애플리케이션의 페이지 전환과 네비게이션을 쉽게 설정할 수 있습니다. 주요 용도는 다음과 같습니다.

- 탭 네비게이션 : 여러 탭을 사용하여 애플리케이션의 주요 섹션을 정의할 수 있습니다. 예를 들어, 홈, 설정, 프로필 등의 탭을 만들 수 있습니다.
- 플라이아웃 메뉴 :  플라이아웃 메뉴를 사용하여 애플리케이션의 다양한 페이지로 이동할 수 있습니다.  햄버거 메뉴 아이콘을 클릭하면 메뉴가 나타나고, 사용자는 원하는 페이지로 이동할 수 있습니다.
- 페이지 라우팅 : AppShell을 사용하면 페이지 간의 네비게이션 경로를 정의할 수 있습니다.  코드상에서 URL 라우팅을 통해 특정 페이지로 쉽게 이동할 수 있습니다.

이번에는 AppShell.xaml 파일을 수정해서 여러개의 페이지를  이동할 수 있도록 지원하도록 해보겠습니다.

### Flyout Menu

**Flyout**은 모바일 앱에서 흔히 볼 수 있는 "햄버거 메뉴"처럼, 좌측 또는 우측에서 슬라이딩으로 나타나는 메뉴입니다. 이 방식은 앱의 주요 페이지로 쉽게 이동할 수 있도록 하는 네비게이션 패턴입니다.

[AppShell.cs]

```xml
<?xml version="1.0" encoding="UTF-8" ?>
<Shell
    x:Class="MyFirstMauiApp.AppShell"
    xmlns="http://schemas.microsoft.com/dotnet/2021/maui"
    xmlns:x="http://schemas.microsoft.com/winfx/2009/xaml"
    xmlns:local="clr-namespace:MyFirstMauiApp"
    Title="MyFirstMauiApp"
    Shell.FlyoutBehavior="Flyout">

    <ShellContent
        Title="Home"
        ContentTemplate="{DataTemplate local:MainPage}"
        Route="MainPage" />
    <ShellContent
        Title="NewPage1"
        ContentTemplate="{DataTemplate local:NewPage1}"
        Route="NewPage1" />
    <ShellContent
        Title="NewPage2"
        ContentTemplate="{DataTemplate local:NewPage2}"
        Route="NewPage2" />
    <ShellContent
        Title="NewPage3"
        ContentTemplate="{DataTemplate local:NewPage3}"
        Route="NewPage3" />

</Shell>
```

기존에 설정되어 있던 FlyoutBehavior 속성을 "Flyout" 으로 변경하고 추가할 페이지를 ShellContent 태그를 이용하여 추가하면 됩니다.

```
    Shell.FlyoutBehavior="Flyout"
```

![](/images/blog/2024/10/flyout-menu.png)

상단의 메뉴 버튼을 누르면 페이지 목록을 확인할 수 있습니다.

![](/images/blog/2024/10/open-flyout-menu.png)

### Tab Navigation

MAUI(Multi-platform App UI)에서 **TabPage**는 앱 내에서 여러 페이지를 탭(Tab)으로 나눠 표시하고, 사용자가 각 탭을 클릭해서 페이지 간에 빠르게 이동할 수 있는 탐색 방식입니다. 이는 탭바(Tabs)를 사용해 여러 관련된 콘텐츠를 하나의 화면에서 그룹화하는 데 유용하며, 모바일 및 데스크톱 환경 모두에서 널리 사용됩니다.

```xml
<?xml version="1.0" encoding="UTF-8" ?>
<Shell
    x:Class="MyFirstMauiApp.AppShell"
    xmlns="http://schemas.microsoft.com/dotnet/2021/maui"
    xmlns:x="http://schemas.microsoft.com/winfx/2009/xaml"
    xmlns:local="clr-namespace:MyFirstMauiApp"
    Title="MyFirstMauiApp"
    Shell.FlyoutBehavior="Disabled">
    <TabBar>
        <Tab Title="Main Page" Icon="dotnet_bot.png">
            <ShellContent
                Title="Home"
                ContentTemplate="{DataTemplate local:MainPage}"
                Icon="dotnet_bot.png"
                Route="MainPage" />
        </Tab>
        <Tab Title="New Page1" Icon="dotnet_bot.png">
            <ShellContent
                Title="NewPage1"
                ContentTemplate="{DataTemplate local:NewPage1}"
                Route="NewPage1" />
        </Tab>
        <Tab Title="New Page2" Icon="dotnet_bot.png">
            <ShellContent
                Title="NewPage2"
                ContentTemplate="{DataTemplate local:NewPage2}"
                Route="NewPage2" />
        </Tab>
        <Tab Title="New Page3" Icon="dotnet_bot.png">
            <ShellContent
                Title="NewPage3"
                ContentTemplate="{DataTemplate local:NewPage3}"
                Route="NewPage3" />
        </Tab>
    </TabBar>
</Shell>
```

Tab Navigation 방식을 이용하려면, TabBar 태크를 이용해서 페이지를 설정하면 됩니다. 아래는 실행 화면입니다.

![](/images/blog/2024/10/tab-menu.png)
