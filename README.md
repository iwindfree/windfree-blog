# 바람부는 자유 (ws.blog)

Next.js 15 App Router 기반 개인 기술 블로그.

## Tech Stack

- **Framework**: Next.js 15 (App Router, React 19)
- **Styling**: Tailwind CSS v4, `@tailwindcss/typography`
- **Markdown**: unified + remark + rehype 파이프라인
- **Syntax Highlighting**: Shiki (dual theme — light/dark)
- **Search**: Pagefind (정적 검색 인덱스)
- **OG Image**: `@vercel/og`
- **Validation**: Zod (frontmatter 스키마)
- **Deploy**: Vercel

## Getting Started

```bash
pnpm install
pnpm dev
```

http://localhost:3000 에서 확인.

## Scripts

| Command | Description |
|---------|-------------|
| `pnpm dev` | 개발 서버 (Turbopack) |
| `pnpm build` | 프로덕션 빌드 + Pagefind 인덱싱 |
| `pnpm start` | 프로덕션 서버 |
| `pnpm lint` | ESLint |
| `pnpm format` | Prettier 포맷팅 |

## Project Structure

```
src/
├── app/                   # Next.js App Router
│   ├── layout.tsx         # Root layout (메타, 테마, 폰트)
│   ├── page.tsx           # 홈
│   ├── blog/[slug]/       # 블로그 상세
│   ├── tags/[tag]/        # 태그별 포스트
│   ├── categories/[category]/
│   ├── archives/          # 연도별 아카이브
│   ├── search/            # Pagefind 검색
│   ├── about/
│   ├── og/[slug]/route.tsx  # OG 이미지 생성
│   ├── rss.xml/route.ts     # RSS 피드
│   ├── robots.ts
│   └── sitemap.ts
├── components/            # React 컴포넌트
├── data/blog/             # 마크다운 포스트
├── lib/
│   ├── posts.ts           # 포스트 로딩 (gray-matter + fs)
│   ├── mdx.ts             # 마크다운 → HTML 렌더링
│   └── schema.ts          # Zod frontmatter 스키마
├── utils/                 # 정렬, 필터, 슬러그 유틸리티
├── styles/
│   ├── global.css
│   └── typography.css
└── config.ts              # 사이트 설정
```

## 새 포스트 작성 가이드

### 파일 생성

`src/data/blog/my-new-post.md` 경로에 마크다운 파일을 생성한다.

### Frontmatter 템플릿

```markdown
---
title: "글 제목"
author: iwindfree
pubDatetime: 2026-02-18T10:00:00Z
slug: "url-slug-name"
category: "카테고리"
tags: ["tag1", "tag2"]
description: "글 요약 설명"
---
```

### 필수 필드

| 필드 | 타입 | 설명 |
|------|------|------|
| `title` | string | 글 제목 |
| `pubDatetime` | date | 발행일시 (ISO 8601) |
| `description` | string | 글 요약 설명 |

### 선택 필드

| 필드 | 타입 | 기본값 | 설명 |
|------|------|--------|------|
| `author` | string | `iwindfree` | 작성자 |
| `slug` | string | 파일명 | URL 경로 |
| `category` | enum | - | 카테고리 (아래 허용 값 참조) |
| `tags` | string[] | `["others"]` | 태그 목록 |
| `featured` | boolean | - | 메인 페이지 상단 노출 |
| `draft` | boolean | - | `true`이면 비공개 |
| `series` | string | - | 시리즈 이름 |
| `seriesOrder` | number | - | 시리즈 내 순서 |
| `ogImage` | string | - | OG 이미지 경로 |
| `canonicalURL` | string | - | 원본 URL (중복 게시 시) |
| `modDatetime` | date | - | 수정일시 |

### category 허용 값

`MAUI 기본` · `MAUI 활용` · `IT 잡썰` · `LLM Engineering` · `AI Agent Engineering` · `RUST` · `JAVA BCI` · `프로그래밍 노트`

## 이미지 가이드

이미지는 `public/images/blog/<글-폴더명>/` 디렉토리에 저장하고, 본문에서 절대 경로로 참조한다.

```markdown
![설명](/images/blog/java-bci/javaagent.png)
```

디렉토리 구조 예시:

```
public/images/blog/
├── java-bci/
│   ├── java_class_loading.png
│   └── javaagent.png
└── rust-stack-heap/
    └── ...
```

## License

MIT
