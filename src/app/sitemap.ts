import type { MetadataRoute } from "next";
import { getAllPosts } from "@/lib/posts";
import getSortedPosts from "@/utils/getSortedPosts";
import getUniqueTags from "@/utils/getUniqueTags";
import getUniqueCategories from "@/utils/getUniqueCategories";
import { SITE } from "@/config";

export default function sitemap(): MetadataRoute.Sitemap {
  const baseUrl = SITE.website.replace(/\/$/, "");
  const posts = getAllPosts();
  const sortedPosts = getSortedPosts(posts);
  const tags = getUniqueTags(posts);
  const categories = getUniqueCategories(posts);

  const staticPages: MetadataRoute.Sitemap = [
    { url: baseUrl, lastModified: new Date(), changeFrequency: "daily" },
    { url: `${baseUrl}/blog`, lastModified: new Date(), changeFrequency: "daily" },
    { url: `${baseUrl}/about`, changeFrequency: "monthly" },
    { url: `${baseUrl}/tags`, changeFrequency: "weekly" },
    { url: `${baseUrl}/categories`, changeFrequency: "weekly" },
    ...(SITE.showArchives
      ? [{ url: `${baseUrl}/archives`, changeFrequency: "weekly" as const }]
      : []),
    { url: `${baseUrl}/search`, changeFrequency: "monthly" },
  ];

  const postPages: MetadataRoute.Sitemap = sortedPosts.map((post) => ({
    url: `${baseUrl}/blog/${post.slug}`,
    lastModified: new Date(post.data.modDatetime ?? post.data.pubDatetime),
    changeFrequency: "weekly",
  }));

  const tagPages: MetadataRoute.Sitemap = tags.map(({ tag }) => ({
    url: `${baseUrl}/tags/${tag}`,
    changeFrequency: "weekly",
  }));

  const categoryPages: MetadataRoute.Sitemap = categories.map((category) => ({
    url: `${baseUrl}/categories/${encodeURIComponent(category)}`,
    changeFrequency: "weekly",
  }));

  return [...staticPages, ...postPages, ...tagPages, ...categoryPages];
}
