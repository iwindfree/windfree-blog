import Link from "next/link";
import type { BlogPost } from "@/lib/posts";
import { getFirstImage } from "@/utils/getFirstImage";
import Datetime from "./Datetime";

interface CardBlogProps {
  post: BlogPost;
}

export default function CardBlog({ post }: CardBlogProps) {
  const { title, description, category, tags, ogImage, ...datetimeProps } =
    post.data;
  const thumbnail = ogImage ?? getFirstImage(post.body);

  return (
    <Link
      href={`/blog/${post.slug}`}
      className="card-transition group flex flex-col overflow-hidden rounded-xl border border-border bg-card-bg shadow-[0_2px_16px_rgba(0,0,0,0.06)] hover:scale-[1.02] hover:shadow-[0_8px_32px_rgba(0,0,0,0.10)]"
    >
      <div className="aspect-[16/9] overflow-hidden">
        {thumbnail ? (
          <img
            src={thumbnail}
            alt={title}
            className="h-full w-full object-cover transition-transform duration-500 group-hover:scale-105"
            loading="lazy"
          />
        ) : (
          <div className="flex h-full w-full items-center justify-center bg-muted">
            <span className="text-lg font-semibold text-foreground/30">
              {category ?? "Blog"}
            </span>
          </div>
        )}
      </div>
      <div className="flex flex-1 flex-col p-5">
        {category && (
          <span className="mb-2 text-xs font-semibold uppercase tracking-wider text-accent-gold">
            {category}
          </span>
        )}
        <h3 className="mb-2 text-lg font-semibold leading-snug transition-colors group-hover:text-accent">
          {title}
        </h3>
        <Datetime {...datetimeProps} className="mb-2 text-sm opacity-60" />
        <p className="mb-3 line-clamp-2 text-sm opacity-75">{description}</p>
        {tags && tags.length > 0 && (
          <div className="mt-auto flex flex-wrap gap-1.5">
            {tags.slice(0, 3).map((tag) => (
              <span
                key={tag}
                className="rounded-full bg-muted px-2.5 py-0.5 text-xs font-medium opacity-75"
              >
                {tag}
              </span>
            ))}
            {tags.length > 3 && (
              <span className="rounded-full bg-muted px-2.5 py-0.5 text-xs font-medium opacity-50">
                +{tags.length - 3}
              </span>
            )}
          </div>
        )}
      </div>
    </Link>
  );
}
