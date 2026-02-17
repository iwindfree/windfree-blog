import Link from "next/link";
import type { BlogPost } from "@/lib/posts";

interface SeriesNavigationProps {
  currentPost: BlogPost;
  allPosts: BlogPost[];
}

export default function SeriesNavigation({
  currentPost,
  allPosts,
}: SeriesNavigationProps) {
  const series = currentPost.data.series;
  if (!series) return null;

  const seriesPosts = allPosts
    .filter((p) => p.data.series === series)
    .sort(
      (a, b) => (a.data.seriesOrder ?? 0) - (b.data.seriesOrder ?? 0)
    );

  if (seriesPosts.length <= 1) return null;

  const currentIndex = seriesPosts.findIndex(
    (p) => p.id === currentPost.id
  );
  const prevPost = currentIndex > 0 ? seriesPosts[currentIndex - 1] : null;
  const nextPost =
    currentIndex < seriesPosts.length - 1
      ? seriesPosts[currentIndex + 1]
      : null;

  return (
    <div className="my-8 rounded-lg border border-border bg-muted/30 p-4">
      <h3 className="mb-3 text-lg font-semibold">
        {series} ({currentIndex + 1}/{seriesPosts.length})
      </h3>
      <ol className="mb-4 list-inside list-decimal space-y-1 text-sm">
        {seriesPosts.map((post, i) => (
          <li
            key={post.id}
            className={
              i === currentIndex ? "font-bold text-accent" : "opacity-75"
            }
          >
            {i === currentIndex ? (
              <span>{post.data.title}</span>
            ) : (
              <Link
                href={`/blog/${post.slug}`}
                className="hover:text-accent hover:underline"
              >
                {post.data.title}
              </Link>
            )}
          </li>
        ))}
      </ol>
      <div className="flex justify-between gap-4">
        {prevPost ? (
          <Link
            href={`/blog/${prevPost.slug}`}
            className="text-sm hover:text-accent"
          >
            &larr; 이전: {prevPost.data.title}
          </Link>
        ) : (
          <span />
        )}
        {nextPost ? (
          <Link
            href={`/blog/${nextPost.slug}`}
            className="text-sm text-end hover:text-accent"
          >
            다음: {nextPost.data.title} &rarr;
          </Link>
        ) : (
          <span />
        )}
      </div>
    </div>
  );
}
