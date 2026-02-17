import Link from "next/link";
import { IconHash } from "./Icons";

interface TagProps {
  tag: string;
  tagName: string;
  size?: "sm" | "lg";
}

export default function Tag({ tag, tagName, size = "lg" }: TagProps) {
  return (
    <li>
      <Link
        href={`/tags/${tag}/`}
        className={`flex items-center gap-0.5 border-b-2 border-dashed border-foreground hover:-mt-0.5 hover:border-accent hover:text-accent focus-visible:border-none focus-visible:text-accent ${size === "sm" ? "text-sm" : "text-lg"}`}
      >
        <IconHash
          className={`opacity-80 ${size === "lg" ? "size-5" : "size-4"}`}
        />
        {tagName}
      </Link>
    </li>
  );
}
