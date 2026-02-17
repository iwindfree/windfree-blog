import type { Metadata } from "next";
import { getAllPosts } from "@/lib/posts";
import getUniqueCategories from "@/utils/getUniqueCategories";
import getPostsByCategory from "@/utils/getPostsByCategory";
import Header from "@/components/Header";
import Footer from "@/components/Footer";
import Breadcrumb from "@/components/Breadcrumb";
import CardBlog from "@/components/CardBlog";

interface Props {
  params: Promise<{ category: string }>;
}

export async function generateStaticParams() {
  const posts = getAllPosts();
  const categories = getUniqueCategories(posts);
  return categories.map((category) => ({ category }));
}

export async function generateMetadata({ params }: Props): Promise<Metadata> {
  const { category } = await params;
  const decoded = decodeURIComponent(category);
  return {
    title: `Category: ${decoded}`,
  };
}

export default async function CategoryPostsPage({ params }: Props) {
  const { category } = await params;
  const decoded = decodeURIComponent(category);
  const posts = getAllPosts();
  const categoryPosts = getPostsByCategory(posts, decoded);

  return (
    <>
      <Header />
      <Breadcrumb />
      <main id="main-content" className="app-layout pb-4">
        <h1 className="text-2xl font-semibold sm:text-3xl">
          Category: <span>{decoded}</span>
        </h1>
        <p className="mt-2 mb-6 italic">
          &quot;{decoded}&quot; 카테고리의 모든 글.
        </p>
        <div className="grid grid-cols-1 gap-6 sm:grid-cols-2 lg:grid-cols-3">
          {categoryPosts.map((post) => (
            <CardBlog key={post.id} post={post} />
          ))}
        </div>
      </main>
      <Footer />
    </>
  );
}
