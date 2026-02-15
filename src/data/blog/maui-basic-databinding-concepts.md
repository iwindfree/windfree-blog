---
title: "[MAUI 기본] 데이터바인딩 - 기본 개념"
author: iwindfree
pubDatetime: 2024-10-15T15:54:57Z
slug: "maui-basic-databinding-concepts"
category: "MAUI 기본"
tags: [".net", "maui", "mobile app"]
description: "데이터 바인딩Data Binding은 UI 요소와 데이터 소스 간의 연결을 설정하여, 데이터가 변경될 때 UI가 자동으로 업데이트되도록 하는 기술입니다. 이를 통해 코드의 유지보수성과 재사용성을 높일 수 있습니다. MAUI 에서 데이터 바인딩은 XAML이나 코드에서"
---

데이터 바인딩(Data Binding)은 UI 요소와 데이터 소스 간의 연결을 설정하여, 데이터가 변경될 때 UI가 자동으로 업데이트되도록 하는 기술입니다. 이를 통해 코드의 유지보수성과 재사용성을 높일 수 있습니다. MAUI 에서 데이터 바인딩은 XAML이나 코드에서 구현할 수 있지만, 코드 비하인드 파일의 크기를 줄이는 데 도움이 되는 XAML에서 정의하는 것이  훨씬 더 일반적입니다. 아래는 간단한 개념도입니다.

![](/images/blog/2024/10/databinding-basic-2.png)

간단한 예제를 통해서 설명드리겠습니다. 우리 예제는 아래와 같이 구성되어 있습니다.

- View : MainPage
- Model : People
- ViewModel: PeopleViewModel

[People.cs]

```csharp
public class People
{
    public string Name { get; set; }
    public int Age { get; set; }
}
```

People 은 모델 클래스의 역할을 수행하며, 간단하게 Name 과 Age 라는 속성을 갖고 있습니다. 이제 ViewModel 을 살펴보겠습니다.

[PeopleViewModel.cs]

```csharp
public partial class PeopleViewModel : ObservableObject
{
    public ObservableCollection<People> Peoples { get; set; } = new ObservableCollection<People>();

    public PeopleViewModel()
    {
        LoadData();
        Title = "DataBind Test Page";
    }

    [ObservableProperty]
    string title;

    private void LoadData()
    {
        Peoples.Add(new People { Name = "홍길동", Age = 25 });
        Peoples.Add(new People { Name = "전우치", Age = 22 });
        Peoples.Add(new People { Name = "피노키오", Age = 28 });
    }
}
```

PeopleViewModel 클래스는 단순하게 People 리스트를 생성해서 ObservableCollection 으로 외부에 제공하는 역할을 수행하고 있습니다.  View 에서는 ViewModel 의 Peoples 속성을 통해서 people 리스트에 접근할 수 있게 되는 거죠.

참고로 MVVM 패턴 구성을 편리하게 할 수 있도록 지원하는 Community Toolkit 을 사용하고 있기 때문에 **[ObservableProperty]** 같은 속성을 사용하여 코드를 간결하게 만들었습니다.

그럼 이제 본격적으로 View 에서 Data Binding 을 통해서 ViewModel 과 어떻게 데이터를 주고 받는지 확인해 보겠습니다. 간단하게 MainPage.xaml 을 아래와 같이 수정하였습니다.

```csharp
<ContentPage
    x:Class="BindingSample.MainPage"
    xmlns="http://schemas.microsoft.com/dotnet/2021/maui"
    xmlns:x="http://schemas.microsoft.com/winfx/2009/xaml"
    xmlns:viewmodel="clr-namespace:BindingSample.ViewModels"
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

BindingSample.ViewModels 라는  네임스페이스를 viewmodel 이라는 접두사로 참조하기 위하여 아래와 같이 네임스페이스를 정의하였습니다.

```
   xmlns:viewmodel="clr-namespace:BindingSample.ViewModels"
```

실제 View 에서 사용할 BindingContext 를 정의해보겠습니다. **BindingContext **는 XAML에서 데이터 바인딩의 소스를 설정하는 속성입니다. 이를 통해 **UI 요소가 데이터 소스와 연결**되어 데이터 바인딩을 수행할 수 있습니다. BindingContext는 주로 페이지, 레이아웃, 또는 개별 컨트롤에 설정됩니다. 데이터바인딩을 위하여 꼭 BindingContext 를 사용할 필요는 없지만, 가장 일반적으로 사용되는 방식입니다.

```
<ContentPage.BindingContext>
    <viewmodel:PeopleViewModel />
</ContentPage.BindingContext>
```

XAML 에서는 위와 같이 설정하여 페이지 레벨에서의 BindingContext 를 지정할 수 있습니다. 여기서 **페이지 레벨 **이라고 한 것은 페이지의 하부요소에서도 상속의 개념으로 페이지 레벨에서 설정한 BindingContext 를 사용할 수 있다는 것입니다. BindingContext 는 페이지 레벨이 아닌 하위 레벨에서도 설정할 수 있다는 점을 기억하시기 바랍니다.  위와 같이 XAML 내부에서도 설정할 수 있고 코드 비하인드 파일 (*.xaml.cs) 에서도 설정할 수 있습니다.  좀 더 복잡한 로직으로 BindingContext 설정이 필요한 경우 사용하실 수 있습니다.

```csharp
 public MainPage()
 {
     InitializeComponent();
     BindingContext = new PeopleViewModel();
 }
```

이제 실제로 BindingContext 를 이용하여 UI 요소에 데이터를 출력해 보겠습니다.

```xml
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
```

위의 XAML 코드에 대해서 설명하겠습니다,

- 데이터 바인딩 : **ItemsSource="{Binding Peoples}" **는 CollectionView의 데이터 소스를 PeopleViewModel의 Peoples 컬렉션에 바인딩합니다. BindingContext 를  PeopleViewModel 로 정의했기 때문에 직접 PeopleViewModel 의 Peoples 라는 속성과 바인딩할 수 있습니다.
- 컬렉션데이터 표시 : CollectonView 는 Peoples 컬렉션의 개별 항목을 렌더링하여, DataTemplate 을 사용하여 각 항목의 레이아웃을 정의합니다.

여기서 한가지 의문이 생깁니다. 바로 아래 부분인데요.

```xml
<DataTemplate>
    <StackLayout Padding="10" Spacing="5">
        <Label Text="{Binding Name}" />
        <Label Text="{Binding Age}" />
    </StackLayout>
</DataTemplate>
```

Label 의 Text 에 Name 과 Age 라는 속성을 바인딩 시켰습니다. 그러나, 우리가 BindingContext 로 설정한 PeopleViewModel 에는 Name 과 Age 속성이 없습니다. 이는 People 클래스에 있는 속성입니다. 그렇다고 별도로 BindingContext 로 People 클래스를 설정한 적도 없는데 코드는 정상적으로 동작합니다.  결그 이유에 대해서 설명드리겠습니다.

#### ItemSource 설정

CollectionView의 ItemsSource는 현재  PeopleViewModel의 Peoples 컬렉션으로 설정되어 있습니다. Peoples 는 People 을 개별 요소로 갖고 있는 Collection 입니다.

```xml
<CollectionView ItemsSource="{Binding Peoples}" SelectionMode="Single">
```

#### DataTemplate 내 바인딩

그리고, CollectionView 의 개별 항목을 어떻게 표시할지를 지정하는 DataTemplate 내의 각 항목은 People 객체입니다. MAUI 에서는 자동으로 이러한 경우 People 클래스의 Name과 Age 속성에 바인딩할 수 있도록 해주고 있습니다.

```xml
<DataTemplate>
    <StackLayout Padding="10" Spacing="5">
        <Label Text="{Binding Name}" />
        <Label Text="{Binding Age}" />
    </StackLayout>
</DataTemplate>
```

DataTemplate 내의 각 항목은 People 객체이므로, People 객체의 Name과 Age 속성에 바인딩할 수 있는 것입니다. 별도로 People 클래스를 명시적으로 참조하지 않아도, BindingContext 와 ItemsSource 설정 덕분에 DataTemplate 내에서 People 객체의 속성에 바인딩할 수 있는 것입니다.  아래는 실행한 화면입니다.

![](/images/blog/2024/10/binding-basic-result.png)
*실행 화면*
