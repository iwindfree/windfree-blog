import Link from "next/link";
import type { BlogPost } from "@/lib/posts";
import Datetime from "./Datetime";

interface CardSimpleProps {
  post: BlogPost;
}

export default function CardSimple({ post }: CardSimpleProps) {
  const { title, description, ...datetimeProps } = post.data;

  return (
    <li className="my-6">
      <Link
        href={`/blog/${post.slug}`}
        className="inline-block text-lg font-medium text-accent decoration-dashed underline-offset-4 hover:underline focus-visible:no-underline focus-visible:underline-offset-0"
      >
        <h2>{title}</h2>
      </Link>
      <Datetime {...datetimeProps} />
      <p>{description}</p>
    </li>
  );
}
