import type { Metadata } from "next";
import { getAllPosts } from "@/lib/posts";
import getUniqueTags from "@/utils/getUniqueTags";
import Header from "@/components/Header";
import Footer from "@/components/Footer";
import Breadcrumb from "@/components/Breadcrumb";
import Tag from "@/components/Tag";

export const metadata: Metadata = {
  title: "Tags",
};

export default function TagsPage() {
  const posts = getAllPosts();
  const tags = getUniqueTags(posts);

  return (
    <>
      <Header />
      <Breadcrumb />
      <main id="main-content" className="app-layout pb-4">
        <h1 className="text-2xl font-semibold sm:text-3xl">Tags</h1>
        <p className="mt-2 mb-6 italic">All the tags used in posts.</p>
        <ul className="flex flex-wrap gap-6">
          {tags.map(({ tag, tagName }) => (
            <Tag key={tag} tag={tag} tagName={tagName} />
          ))}
        </ul>
      </main>
      <Footer />
    </>
  );
}
