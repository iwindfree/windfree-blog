import type { Metadata } from "next";
import Link from "next/link";
import { notFound } from "next/navigation";
import { getAllPosts, getPostBySlug } from "@/lib/posts";
import getSortedPosts from "@/utils/getSortedPosts";
import { slugifyStr } from "@/utils/slugify";
import { renderMarkdown } from "@/lib/mdx";
import Header from "@/components/Header";
import Footer from "@/components/Footer";
import BackButton from "@/components/BackButton";
import Tag from "@/components/Tag";
import Datetime from "@/components/Datetime";
import ShareLinks from "@/components/ShareLinks";
import BackToTopButton from "@/components/BackToTopButton";
import SeriesNavigation from "@/components/SeriesNavigation";
import PostDetailClient from "@/components/PostDetailClient";
import { IconChevronLeft, IconChevronRight } from "@/components/Icons";
import { SITE } from "@/config";

interface Props {
  params: Promise<{ slug: string }>;
}

export async function generateStaticParams() {
  const posts = getAllPosts().filter((p) => !p.data.draft);
  return posts.map((post) => ({ slug: post.slug }));
}

export async function generateMetadata({ params }: Props): Promise<Metadata> {
  const { slug } = await params;
  const post = getPostBySlug(slug);
  if (!post) return {};

  const ogImage = post.data.ogImage
    ? post.data.ogImage
    : SITE.dynamicOgImage
      ? `/og/${slug}/og.png`
      : undefined;

  return {
    title: post.data.title,
    description: post.data.description,
    openGraph: {
      title: post.data.title,
      description: post.data.description,
      type: "article",
      publishedTime: post.data.pubDatetime.toISOString(),
      ...(post.data.modDatetime && {
        modifiedTime: post.data.modDatetime.toISOString(),
      }),
      ...(ogImage && { images: [ogImage] }),
    },
    twitter: {
      card: "summary_large_image",
      title: post.data.title,
      description: post.data.description,
      ...(ogImage && { images: [ogImage] }),
    },
  };
}

export default async function BlogPostPage({ params }: Props) {
  const { slug } = await params;
  const post = getPostBySlug(slug);
  if (!post) notFound();

  const allPosts = getAllPosts();
  const sortedPosts = getSortedPosts(allPosts);
  const content = await renderMarkdown(post.body);

  const allPostsList = sortedPosts.map((p) => ({
    id: p.id,
    slug: p.slug,
    title: p.data.title,
  }));

  const currentPostIndex = allPostsList.findIndex((a) => a.id === post.id);
  const prevPost =
    currentPostIndex !== 0 ? allPostsList[currentPostIndex - 1] : null;
  const nextPost =
    currentPostIndex !== allPostsList.length - 1
      ? allPostsList[currentPostIndex + 1]
      : null;

  const {
    title,
    pubDatetime,
    modDatetime,
    timezone,
    tags,
    category,
  } = post.data;

  return (
    <>
      <Header />
      <BackButton />
      <main
        id="main-content"
        className={`app-layout pb-12 ${!SITE.showBackButton ? "mt-8" : ""}`}
        data-pagefind-body
      >
        {category && (
          <Link
            href={`/categories/${encodeURIComponent(category)}`}
            className="mb-2 inline-block rounded-full bg-accent-gold/10 px-3 py-1 text-xs font-semibold text-accent-gold transition-colors hover:bg-accent-gold/20"
          >
            {category}
          </Link>
        )}
        <h1 className="inline-block text-2xl font-bold text-accent sm:text-3xl">
          {title}
        </h1>
        <div className="my-2 flex items-center gap-2">
          <Datetime
            pubDatetime={pubDatetime}
            modDatetime={modDatetime}
            timezone={timezone}
            size="lg"
          />
        </div>
        {post.data.series && (
          <SeriesNavigation currentPost={post} allPosts={sortedPosts} />
        )}
        <article
          id="article"
          className="app-prose mt-8 w-full max-w-app"
          dangerouslySetInnerHTML={{ __html: content }}
        />

        <hr className="my-8 border-dashed" />

        <ul className="mt-4 mb-8 flex flex-wrap gap-4 sm:my-8">
          {tags.map((tag) => (
            <Tag
              key={tag}
              tag={slugifyStr(tag)}
              tagName={tag}
              size="sm"
            />
          ))}
        </ul>

        <BackToTopButton />
        <ShareLinks />

        <hr className="my-6 border-dashed" />

        {/* Previous/Next Post Buttons */}
        <div
          data-pagefind-ignore
          className="grid grid-cols-1 gap-6 sm:grid-cols-2"
        >
          {prevPost && (
            <Link
              href={`/blog/${prevPost.slug}`}
              className="flex w-full gap-1 hover:opacity-75"
            >
              <IconChevronLeft className="inline-block flex-none rtl:rotate-180" />
              <div>
                <span>Previous Post</span>
                <div className="text-sm text-accent/85">
                  {prevPost.title}
                </div>
              </div>
            </Link>
          )}
          {nextPost && (
            <Link
              href={`/blog/${nextPost.slug}`}
              className="flex w-full justify-end gap-1 text-end hover:opacity-75 sm:col-start-2"
            >
              <div>
                <span>Next Post</span>
                <div className="text-sm text-accent/85">
                  {nextPost.title}
                </div>
              </div>
              <IconChevronRight className="inline-block flex-none rtl:rotate-180" />
            </Link>
          )}
        </div>
      </main>
      <Footer />
      <PostDetailClient />
    </>
  );
}
