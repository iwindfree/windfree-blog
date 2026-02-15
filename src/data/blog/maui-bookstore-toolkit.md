---
title: "[MAUI 활용] BookStore 만들기 (2) - Community Toolkit 설치"
author: iwindfree
pubDatetime: 2024-10-15T16:21:04Z
slug: "maui-bookstore-toolkit"
category: "MAUI 활용"
tags: [".net", "maui", "mobile app"]
description: "MVVM 패턴을 사용하기 위해서는 MVVM Toolkit 라이브러리를 설치하는 것이 좋습니다. 물론, 이러한 라이브러리 없이 모두 직접 코딩을 통해서 구현을 할 수도 있지만, 코딩해야 할 양도 만만치 않고 신경써야 할 부분도 많습니다. 다행히 시중에는 MAUI 를"
series: "MAUI BookStore 만들기"
seriesOrder: 2
---

MVVM 패턴을 사용하기 위해서는 MVVM Toolkit 라이브러리를 설치하는 것이 좋습니다. 물론, 이러한 라이브러리 없이 모두 직접 코딩을 통해서 구현을 할 수도 있지만, 코딩해야 할 양도 만만치 않고 신경써야 할 부분도 많습니다. 다행히 시중에는  MAUI 를 통해서 앱을 개발할 때 MVVM 패턴을 쉽게 구현할 수 있도록 도와주는 라이브러리들이 존재합니다. 여기에서는 그 중에서 Microsoft 가 제공하는 "Community ToolKit" 을 설치할 겁니다.

.NET Community Toolkit은 .NET 개발자들이 애플리케이션을 더 쉽게 개발할 수 있도록 도와주는 유틸리티와 컨트롤 모음입니다. 이 툴킷은 다양한 기능을 제공하며, 특히 MVVM 패턴을 사용하는 애플리케이션 개발에 유용합니다. Community Toolkit은 여러 플랫폼에서 사용할 수 있으며, 주로 WPF, UWP, Xamarin, MAUI와 같은 XAML 기반 프레임워크에서 많이 사용됩니다.

## 필요 패키지 설치하기

ViauslStudio 에서 프로젝트를 선택하고 우측 마우스를 클릭하면 보이는 컨텍스트 메뉴에서  "솔루션용 NugetPackage 관리" 를 선택합니다. 패키지 검색란에 "mvvm" 라고 검색을 하면 Community ToolKit 을 확인할 수 있습니다. 해당 패키지를 설치하면 됩니다. 향후에 해당 패키지를 사용하여 개발하는 방법에 대해서 설명하겠습니다.

![](/images/blog/2024/10/toolkit-1024x719.png)

* 저는 CommunityToolkit 버전을 8.2.2 버전으로 설치하였습니다. 이 글을 쓰는 시점에서 안정화된 최신 버전은 8.3.2 였지만 설치된 .NET 버전(8.0.402) 과 충돌을 일으키는 것 같아서 버전을 한단계 낮춰서 사용했습니다.
