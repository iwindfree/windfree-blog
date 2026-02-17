import type { Metadata } from "next";
import { getAllPosts } from "@/lib/posts";
import getSortedPosts from "@/utils/getSortedPosts";
import getUniqueCategories from "@/utils/getUniqueCategories";
import Header from "@/components/Header";
import Footer from "@/components/Footer";
import CategoryFilter from "@/components/CategoryFilter";
import { SITE } from "@/config";

export const metadata: Metadata = {
  title: "Blog",
};

export default function BlogPage() {
  const posts = getAllPosts();
  const sortedPosts = getSortedPosts(posts);
  const categories = getUniqueCategories(posts);

  return (
    <>
      <Header />
      <main id="main-content" className="app-layout-wide py-12">
        <h1 className="mb-2 text-3xl font-bold">Blog</h1>
        <p className="mb-8 opacity-60">함께하는 프로그래밍과 IT 이야기</p>
        <CategoryFilter posts={sortedPosts} categories={categories} />
      </main>
      <Footer />
    </>
  );
}
