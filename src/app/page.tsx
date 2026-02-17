import Link from "next/link";
import { getAllPosts } from "@/lib/posts";
import getSortedPosts from "@/utils/getSortedPosts";
import Header from "@/components/Header";
import Footer from "@/components/Footer";
import CardHome from "@/components/CardHome";
import FadeIn from "@/components/FadeIn";
import { IconArrowRight } from "@/components/Icons";

export default function HomePage() {
  const posts = getAllPosts();
  const sortedPosts = getSortedPosts(posts);
  const recentPosts = sortedPosts.slice(0, 5);
  const featuredPost = recentPosts[0];
  const listPosts = recentPosts.slice(1);

  return (
    <>
      <Header />
      <main id="main-content">
        {/* Hero Section */}
        <section className="relative h-[75vh] min-h-[480px] max-h-[720px] overflow-hidden">
          <div
            className="ken-burns absolute inset-0 bg-cover bg-center"
            style={{
              backgroundImage: "url('/images/hero/windfree-hero.jpg')",
            }}
          />
          <div className="absolute inset-0 bg-gradient-to-t from-[#0D1B2A]/60 via-transparent to-transparent" />
          <div className="app-layout relative flex h-full flex-col items-start justify-end pb-16">
            <span className="mb-4 text-xs font-semibold uppercase tracking-[0.2em] text-accent-gold">
              Notes for a Some day
            </span>
            <h1 className="mb-2 text-4xl font-bold text-white sm:text-5xl lg:text-6xl">
              WINDFREE&apos;S BLOG
            </h1>
            <p className="mb-1 text-xl font-medium text-white/90">
              computer programmer
            </p>
            <p className="mb-8 max-w-xl text-base text-white/70">
              함께하는 프로그래밍과 IT 이야기
            </p>
            <Link
              href="/blog"
              className="inline-flex items-center gap-2 rounded-lg bg-accent-gold px-6 py-3 font-semibold text-white transition-all hover:brightness-110 hover:shadow-lg"
            >
              블로그 보기
              <IconArrowRight className="inline-block" />
            </Link>
          </div>
        </section>

        {/* Intro Section */}
        <FadeIn>
          <section className="bg-muted py-16">
            <div className="mx-auto max-w-3xl px-4 text-center">
              <h2 className="text-2xl font-bold text-accent">
                함께하는 프로그래밍과 IT 이야기
              </h2>
              <p className="mt-4 leading-relaxed text-foreground/80 [word-break:keep-all]">
                이 블로그는 제가 프로그래밍과 IT 서비스에 대한 생각을 기록해 두기
                위한 공간입니다. 기술은 끊임없이 발전하고 변화하지만, 그 과정에서
                얻는 경험과 지식은 언제나 소중합니다. 프로그램 개발, IT 서비스,
                그리고 일상 속에서 마주치는 다양한 기술적 고민들을 다루며, 모두가
                쉽게 이해할 수 있도록 풀어보려 합니다.
              </p>
              <img
                src="/images/windfree-logo.jpeg"
                alt="windfree logo"
                className="mx-auto mt-8 w-full max-w-2xl mix-blend-multiply dark:mix-blend-screen dark:invert"
                loading="lazy"
              />
            </div>
          </section>
        </FadeIn>

        {/* Recent Posts — Magazine Layout */}
        <FadeIn>
          <section className="app-layout-wide py-20">
            <div className="mb-10 text-center">
              <h2 className="text-2xl font-bold">최근 글</h2>
              <p className="mt-2 text-sm opacity-60">
                새로 올라온 포스트를 확인하세요
              </p>
            </div>
            <div className="grid grid-cols-1 gap-8 lg:grid-cols-5">
              {featuredPost && (
                <CardHome post={featuredPost} variant="featured" />
              )}
              <div className="flex flex-col lg:col-span-2">
                {listPosts.map((post) => (
                  <CardHome key={post.id} post={post} variant="list" />
                ))}
              </div>
            </div>
            <div className="mt-10 text-center">
              <Link
                href="/blog"
                className="inline-flex items-center gap-1 rounded-lg border border-accent px-6 py-2.5 text-sm font-medium text-accent transition-all hover:bg-accent hover:text-white"
              >
                모든 글 보기
                <IconArrowRight className="inline-block h-4 w-4" />
              </Link>
            </div>
          </section>
        </FadeIn>
      </main>
      <Footer />
    </>
  );
}
