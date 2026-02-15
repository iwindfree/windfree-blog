---
title: "[MAUI 활용] BookStore 만들기 (3) - Model 과 Service 만들기"
author: iwindfree
pubDatetime: 2024-10-15T16:23:13Z
slug: "maui-bookstore-model"
category: "MAUI 활용"
tags: [".net", "maui", "mobile app"]
description: "이제 프로젝트를 생성했고 본격적으로 BookStore 앱을 개발하기 위한 작업을 진행하겠습니다. BookStore 앱은 도서 정보에 대한 간단한 CRUD 기능을 제공할 예정입니다. 이를 위해서 간단히 책에 대한 정보를 갖고 있는 Model 클래스를 생성하겠습니다."
series: "MAUI BookStore 만들기"
seriesOrder: 3
---

이제 프로젝트를 생성했고 본격적으로 BookStore 앱을 개발하기 위한 작업을 진행하겠습니다. BookStore 앱은 도서 정보에 대한 간단한 CRUD 기능을 제공할 예정입니다. 이를 위해서 간단히 책에 대한 정보를 갖고 있는 Model 클래스를 생성하겠습니다.

```csharp
  public class Book
  {
      /// <summary>
      /// Gets or sets the ID of the book.
      /// </summary>
      public int Id { get; set; }

      /// <summary>
      /// Gets or sets the title of the book.
      /// </summary>
      public string BookTitle { get; set; }

      /// <summary>
      /// Gets or sets the author of the book.
      /// </summary>
      public string Author { get; set; }

      /// <summary>
      /// Gets or sets the genre of the book. (책 장르)
      /// </summary>
      public string Genre { get; set; }

      /// <summary>
      /// Gets or sets the publication year of the book.
      /// </summary>
      public int Year { get; set; }
  }
```

그리고, 서점을 표현하는 Model 클래스도 생성하겠습니다.

```csharp
    public class BookStore
    {
        // Unique identifier of the bookstore
        public int Id { get; set; }

        // List of books
        public List<Book> Books { get; set; }
    }
```

이제 간단하게 모든 도서 정보를 제공하는 서비스를 생성하겠습니다. Services 폴더 밑에  BookService 라는 클래스를 생성하겠습니다.

```csharp
public class BookService
{
    List<Book> books;
    public BookService()
    {
        InitBook();
    }

    private void InitBook()
    {

        books = new List<Book>
            {
                new Book { Id = 1, BookTitle = "맨발걷기", Author = "작가1", Genre = "건강", Year = 2018 },
                new Book { Id = 2, BookTitle = "Professional C# 6 and .NET Core 1.0", Author = "windfree", Genre = "프로그래밍", Year = 2016 },
                new Book { Id = 3, BookTitle = "제주도 한달살기", Author = "작가2", Genre = "여행", Year = 2013 },
                new Book { Id = 4, BookTitle = "자연치유 연구", Author = "작가3", Genre = "건강", Year = 2010 },
                new Book { Id = 5, BookTitle = "Professional C# 3.0", Author = "windfree", Genre = "프로그래밍", Year = 2008 },
                new Book { Id = 6, BookTitle = "유럽여행 가이드북", Author = "작가4", Genre = "여행", Year = 2005 },
                new Book { Id = 7, BookTitle = "한달 다이어트 비법 공개", Author = "작가5", Genre = "건강", Year = 2002 },
                new Book { Id = 8, BookTitle = "Professional RUST", Author = "windfree", Genre = "프로그래밍", Year = 2018 },
                new Book { Id = 9, BookTitle = "동남아 일주일 여행", Author = "작가6", Genre = "여행", Year = 2016 },
                new Book { Id = 10, BookTitle = "달리기로 건강해지자", Author = "작가7", Genre = "건강", Year = 2013 },
            };
    }
    public List<Book> GetBooks()
    {
        return books;
    }

    public Book GetBook(int id)
    {
        return books.FirstOrDefault(b => b.Id == id);
    }

    public void AddBook(Book book)
    {
        book.Id = books.Max(b => b.Id) + 1;
        books.Add(book);
    }

    public void DeleteBook(int id)
    {
        var book = books.FirstOrDefault(b => b.Id == id);
        if (book != null)
        {
            books.Remove(book);
        }
    }
}
```

현재는 BookService 에 임의로 도서 정보를 초기화 하고 이에 대한 CRUD 메서드만 존재합니다. 실제 환경에서는 Database 나 API 호출을 통하여 해당 기능을 구현하겠지만  이번 강의에서는 간단하게 메서드 내에서 모든 도서 정보를 생성해서 제공하는 것으로 대신하려 합니다.

BookService 는 나중에 ViewModel 에서 사용하게 됩니다.
