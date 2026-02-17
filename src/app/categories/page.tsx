import type { Metadata } from "next";
import Link from "next/link";
import { getAllPosts } from "@/lib/posts";
import getUniqueCategories from "@/utils/getUniqueCategories";
import getPostsByCategory from "@/utils/getPostsByCategory";
import Header from "@/components/Header";
import Footer from "@/components/Footer";
import Breadcrumb from "@/components/Breadcrumb";

export const metadata: Metadata = {
  title: "Categories",
};

export default function CategoriesPage() {
  const posts = getAllPosts();
  const categories = getUniqueCategories(posts);

  return (
    <>
      <Header />
      <Breadcrumb />
      <main id="main-content" className="app-layout pb-4">
        <h1 className="text-2xl font-semibold sm:text-3xl">Categories</h1>
        <p className="mt-2 mb-6 italic">All categories.</p>
        <ul className="flex flex-wrap gap-6">
          {categories.map((category) => {
            const count = getPostsByCategory(posts, category).length;
            return (
              <li key={category}>
                <Link
                  href={`/categories/${encodeURIComponent(category)}`}
                  className="inline-block rounded-md bg-muted/50 px-4 py-2 font-medium transition-colors hover:bg-accent hover:text-background"
                >
                  {category}
                  <span className="ml-1 text-sm opacity-75">({count})</span>
                </Link>
              </li>
            );
          })}
        </ul>
      </main>
      <Footer />
    </>
  );
}
