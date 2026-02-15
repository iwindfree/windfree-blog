---
title: "[MAUI 기본] MVVM 개요"
author: iwindfree
pubDatetime: 2024-10-15T15:45:03Z
slug: "maui-basoc-mvvm"
category: "MAUI 기본"
tags: [".net", "maui", "mobile app"]
description: "MAUI 에서 자주 사용하는 MVVM 패턴에 대해서 간단하게 설명드리겠습니다. MVVM 패턴 MVVM은 모델Model, 뷰View, 뷰모델ViewModel의 약자로, 이 패턴의 세 가지 구성 요소를 나타냅니다. 아래 다이어그램은 MS 공식 사이트에서 제공하는 MVVM"
---

MAUI 에서 자주 사용하는 MVVM 패턴에 대해서 간단하게 설명드리겠습니다.

## MVVM 패턴

MVVM은 모델(Model), 뷰(View), 뷰모델(ViewModel)의 약자로, 이 패턴의 세 가지 구성 요소를 나타냅니다.  아래 다이어그램은 MS 공식 사이트에서 제공하는 MVVM 패턴의 구성요소를 설명하는 다이어그램입니다.

![](/images/blog/2024/10/mvvm.png)
*출처 : https://learn.microsoft.com/ko-kr/dotnet/architecture/maui/mvvm*

## Model

**MVVM 패턴에 Model**은 **애플리케이션의 데이터**와 **데이터와 관련된** **비즈니스 로직**을 관리하는 구성 요소입니다. **Model** 은 일반적으로 비즈니스 및 유효성 검사 논리와 함께 데이터 모델을 포함하는  도메인 모델을 나타내는 것으로 생각할 수 있는데요. **ViewModel**과 **직접적으로 연결되지만, View와는 분리**되어 있어 **UI에 독립적인** 구조를 유지합니다. 아래는 사용자(User) 를 표현하는 User 모델 클래스입니다.

[Employee.cs]

```csharp
namespace BindingSample.Models
{
    public class Employee
    {
        public string Name { get; set; }
        public int Age { get; set; }      
        // 비즈니스 로직: 사용자 이름 변경
        public void ChangeName(string newName)
        {
            if (string.IsNullOrWhiteSpace(newName))
            {
                throw new ArgumentException("New name cannot be empty.");
            }
            Name = newName;
        }      
        // 비즈니스 로직: 성인 여부 확인
        public bool IsAdult()
        {
            return Age >= 18;
        }
    }   
}
```

## View

View는 사용자에게 보여지는  UI 를  말합니다. XAML 파일로 정의되며, 사용자와의 상호작용을 처리합니다. NET MAUI 애플리케이션에서 뷰는 일반적으로 ContentPage 를 상속받거나 또는 ContentView 를 상속 받습니다. 아래는 간단한 View 클래스 예제입니다.

[MainPage.xaml]

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

## ViewModel

ViewModel은 Model과 View 사이에 위치한 추상화 계층으로, Model 또는 데이터 소스와 통신하여 데이터를 가져오고, 이를 가공하거나 조작하며, View에 필요한 특정 로직과 비즈니스 로직을 구현합니다. 이렇게 하면 View가 직접적으로 Model과 상호 작용하지 않아도 되고, 추가적인 기능이 필요할 때 View는 해당 기능을 구현할 필요가 없습니다. ViewModel이 View 와 Model 사이에서  그 작업을 수행합니다.  View는 ViewModel 을 통하여  데이터 바인딩을 하고, ViewModel 에 정의된 명령(Command) 을 사용하며, ViewModel은 필요에 따라 Model 을 업데이트합니다. 따라서 ViewModel 은 Model과 View 사이에 위치하여 중간 다리 역할을 합니다.

일반적으로 ViewModel 은 View가 바인딩할 수 있는 속성 및 명령을 구현합니다. 또한 변경 알림 이벤트를 사용하여 상태 변경이 있을 때 뷰에 알립니다. Model의 데이터가 업데이트되면, ViewModel은 UI나 뷰에 알리고, View는 자동으로 업데이트됩니다. 이를 통해 앱이 실시간으로 동작하는 느낌을 줍니다.  또한, 비동기 메서드를 지원하므로 I/O 작업이 UI를 차단하지 않습니다. 따라서 앱이 데이터베이스와 통신하고 응답을 기다리는 동안에도 앱이 멈추지 않고 계속 사용할 수 있어 좋은 사용자 경험을 제공합니다.

뷰 모델이 뷰를 사용하여 양방향 데이터 바인딩에 참여하려면 해당 속성이 PropertyChanged 이벤트를 발생시켜야 합니다. 뷰 모델은 INotifyPropertyChanged 인터페이스를 구현하고 속성이 변경되면 PropertyChanged 이벤트를 발생시켜 이 요구 사항을 충족합니다. 컬렉션의 경우 뷰 친화적인 ObservableCollection<T>이 제공됩니다. 이 컬렉션은 컬렉션 변경 알림을 구현하여 개발자가 컬렉션에서 INotifyCollectionChanged 인터페이스를 구현할 필요가 없습니다.

[PeopleViewModel.cs]

```xml
using System.Collections.ObjectModel;
using System.ComponentModel;

namespace BindingSample.ViewModels
{
    public class PeopleViewModel : INotifyPropertyChanged
    {
        public ObservableCollection<Employee> Peoples { get; set; }
        public string Title { get; set; }

        public event PropertyChangedEventHandler PropertyChanged;

        public PeopleViewModel()
        {
            Title = "Employee List";
            Peoples = new ObservableCollection<Employee>
            {
                new Employee { Name = "John Doe", Age = 30 },
                new Employee { Name = "Jane Smith", Age = 25 }
            };
        }

        protected void OnPropertyChanged(string propertyName)
        {
            PropertyChanged?.Invoke(this, new PropertyChangedEventArgs(propertyName));
        }
    }
}
```

## MVVM 패턴의 잇점

- ViewModel은 코드 및 비즈니스 로직 변경을 Model이나 View에 직접 영향을 주지 않고 관리할 수 있도록 도와줍니다. 이는 큰 장점입니다.
- View와 Model의  분리가 더 잘 이루어지므로 ViewModel 과 Model 을 독립적으로 테스트할 수 있습니다. 예를 들어, 비즈니스 로직이 포함된 View를 검증하려면 View를 인스턴스화하고, 비즈니스 로직을 테스트하기 위해 View를 모킹해야 할 수도 있습니다. 하지만 MVVM에서는 서로  분리가 되어 있으므로 ViewModel 을 독립적으로 테스트할 수 있습니다.
- View를 수정해도 ViewModel 에 영향을 미치지 않으며, 팀이 서로 방해받지 않고 작업할 수 있습니다.

이어지는 MAUI 강의에서 MVVM 패턴은 계속 사용될 것이기 때문에 아직 정확히 이해하지 못한다고 해도 너무 걱정하지 않으셔도 됩니다. 사용하다 보면 익숙해지게 됩니다.  개인적으로는 경험상 어떠한 패턴을 사용한다고 할 때 항상 100% 적용하려고 노력할 필요는 없다고 생각됩니다. 어떠한 경우에는 Model 없이 ViewModel 만으로 구현할 수도 있고, 원래 취지와는 다르게 개발이 진행될 수도 있습니다. 프로그래밍에는 아주 정확한 정답이 있지 않기 때문에 너무 패턴에 얽매여서 개발을 진행할 필요는 없다고 생각합니다.
