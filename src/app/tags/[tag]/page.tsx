import type { Metadata } from "next";
import { getAllPosts } from "@/lib/posts";
import getUniqueTags from "@/utils/getUniqueTags";
import getPostsByTag from "@/utils/getPostsByTag";
import Header from "@/components/Header";
import Footer from "@/components/Footer";
import Breadcrumb from "@/components/Breadcrumb";
import CardBlog from "@/components/CardBlog";
import { SITE } from "@/config";

interface Props {
  params: Promise<{ tag: string }>;
}

export async function generateStaticParams() {
  const posts = getAllPosts();
  const tags = getUniqueTags(posts);
  return tags.map(({ tag }) => ({ tag }));
}

export async function generateMetadata({ params }: Props): Promise<Metadata> {
  const { tag } = await params;
  const posts = getAllPosts();
  const tags = getUniqueTags(posts);
  const tagObj = tags.find((t) => t.tag === tag);
  const tagName = tagObj?.tagName ?? tag;

  return {
    title: `Tag: ${tagName}`,
  };
}

export default async function TagPostsPage({ params }: Props) {
  const { tag } = await params;
  const posts = getAllPosts();
  const tags = getUniqueTags(posts);
  const tagObj = tags.find((t) => t.tag === tag);
  const tagName = tagObj?.tagName ?? tag;
  const tagPosts = getPostsByTag(posts, tag);

  return (
    <>
      <Header />
      <Breadcrumb />
      <main id="main-content" className="app-layout pb-4">
        <h1 className="text-2xl font-semibold sm:text-3xl">
          Tag: <span>{tagName}</span>
        </h1>
        <p className="mt-2 mb-6 italic">
          All the articles with the tag &quot;{tagName}&quot;.
        </p>
        <div className="grid grid-cols-1 gap-6 sm:grid-cols-2 lg:grid-cols-3">
          {tagPosts.map((post) => (
            <CardBlog key={post.id} post={post} />
          ))}
        </div>
      </main>
      <Footer />
    </>
  );
}
