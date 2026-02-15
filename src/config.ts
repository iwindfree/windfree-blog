export const SITE = {
  website: "https://ws-blog.vercel.app/",
  author: "iwindfree",
  profile: "https://github.com/iwindfree",
  desc: "함께하는 프로그래밍과 IT 이야기",
  title: "바람부는 자유",
  ogImage: "astropaper-og.jpg",
  lightAndDarkMode: true,
  postPerIndex: 6,
  postPerPage: 8,
  scheduledPostMargin: 15 * 60 * 1000, // 15 minutes
  showArchives: true,
  showBackButton: true,
  editPost: {
    enabled: false,
    text: "Edit page",
    url: "",
  },
  dynamicOgImage: true,
  dir: "ltr",
  lang: "ko",
  timezone: "Asia/Seoul",
} as const;
