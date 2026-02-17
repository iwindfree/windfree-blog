import Link from "next/link";
import type { BlogPost } from "@/lib/posts";
import { getFirstImage } from "@/utils/getFirstImage";
import Datetime from "./Datetime";

interface CardHomeProps {
  post: BlogPost;
  variant?: "featured" | "list";
}

export default function CardHome({ post, variant = "list" }: CardHomeProps) {
  const { title, description, category, ogImage, ...datetimeProps } = post.data;
  const thumbnail = ogImage ?? getFirstImage(post.body);
  const href = `/blog/${post.slug}`;

  if (variant === "featured") {
    return (
      <Link
        href={href}
        className="card-transition group flex flex-col overflow-hidden rounded-xl border border-border bg-card-bg shadow-[0_2px_16px_rgba(0,0,0,0.06)] hover:shadow-[0_8px_32px_rgba(0,0,0,0.10)] lg:col-span-3"
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
            <div className="flex h-full w-full items-center justify-center bg-accent/10">
              <span className="text-xl font-semibold text-accent/40">
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
          <h3 className="mb-2 text-xl font-semibold leading-snug transition-colors group-hover:text-accent">
            {title}
          </h3>
          <Datetime {...datetimeProps} className="mb-2 text-sm opacity-60" />
          <p className="line-clamp-3 text-sm opacity-75">{description}</p>
        </div>
      </Link>
    );
  }

  return (
    <Link
      href={href}
      className="group flex flex-row items-start gap-4 border-b border-border py-4 last:border-b-0"
    >
      <div className="h-20 w-20 shrink-0 overflow-hidden rounded-lg">
        {thumbnail ? (
          <img
            src={thumbnail}
            alt={title}
            className="h-full w-full object-cover"
            loading="lazy"
          />
        ) : (
          <div className="flex h-full w-full items-center justify-center bg-muted">
            <span className="text-sm font-semibold text-foreground/30">
              {category?.charAt(0) ?? "B"}
            </span>
          </div>
        )}
      </div>
      <div className="flex min-w-0 flex-1 flex-col">
        {category && (
          <span className="mb-1 text-xs font-semibold uppercase tracking-wider text-accent-gold">
            {category}
          </span>
        )}
        <h3 className="mb-1 line-clamp-1 text-sm font-semibold leading-snug transition-colors group-hover:text-accent">
          {title}
        </h3>
        <Datetime {...datetimeProps} className="text-xs opacity-50" />
      </div>
    </Link>
  );
}
