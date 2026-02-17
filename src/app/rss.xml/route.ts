import RSS from "rss";
import { getAllPosts } from "@/lib/posts";
import getSortedPosts from "@/utils/getSortedPosts";
import { SITE } from "@/config";

export async function GET() {
  const posts = getAllPosts();
  const sortedPosts = getSortedPosts(posts);

  const feed = new RSS({
    title: SITE.title,
    description: SITE.desc,
    site_url: SITE.website,
    feed_url: `${SITE.website}rss.xml`,
    language: SITE.lang ?? "ko",
  });

  sortedPosts.forEach((post) => {
    feed.item({
      title: post.data.title,
      description: post.data.description,
      url: `${SITE.website}blog/${post.slug}`,
      date: new Date(post.data.modDatetime ?? post.data.pubDatetime),
    });
  });

  return new Response(feed.xml({ indent: true }), {
    headers: {
      "Content-Type": "application/xml; charset=utf-8",
    },
  });
}
