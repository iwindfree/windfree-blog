import type { Metadata } from "next";
import { SITE } from "@/config";
import "@/styles/global.css";

export const metadata: Metadata = {
  title: {
    default: SITE.title,
    template: `%s | ${SITE.title}`,
  },
  description: SITE.desc,
  metadataBase: new URL(SITE.website),
  openGraph: {
    title: SITE.title,
    description: SITE.desc,
    url: SITE.website,
    siteName: SITE.title,
    locale: "ko_KR",
    type: "website",
    images: [
      {
        url: `/${SITE.ogImage}`,
        width: 1200,
        height: 630,
      },
    ],
  },
  twitter: {
    card: "summary_large_image",
    title: SITE.title,
    description: SITE.desc,
    images: [`/${SITE.ogImage}`],
  },
  alternates: {
    types: {
      "application/rss+xml": "/rss.xml",
    },
  },
  icons: {
    icon: "/favicon.svg",
  },
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html dir={SITE.dir} lang={SITE.lang ?? "en"} suppressHydrationWarning>
      <head>
        <meta name="theme-color" content="" />
        <link rel="sitemap" href="/sitemap.xml" />
        <script
          dangerouslySetInnerHTML={{
            __html: `(function(){var t=localStorage.getItem("theme");function g(){if(t)return t;return window.matchMedia("(prefers-color-scheme:dark)").matches?"dark":"light"}var v=g();document.documentElement.setAttribute("data-theme",v);document.documentElement.style.setProperty("--header-bg",v==="dark"?"rgba(13,17,23,0.92)":"rgba(250,251,253,0.92)");document.documentElement.style.setProperty("--menu-bg",v==="dark"?"rgba(13,17,23,0.95)":"rgba(250,251,253,0.95)");})()`,
          }}
        />
      </head>
      <body>
        {children}
      </body>
    </html>
  );
}
