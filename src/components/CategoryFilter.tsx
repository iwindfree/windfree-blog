"use client";

import { useState } from "react";
import type { BlogPost } from "@/lib/posts";
import CardBlog from "./CardBlog";

interface CategoryFilterProps {
  posts: BlogPost[];
  categories: string[];
}

export default function CategoryFilter({
  posts,
  categories,
}: CategoryFilterProps) {
  const [activeCategory, setActiveCategory] = useState("all");

  const filteredPosts =
    activeCategory === "all"
      ? posts
      : posts.filter((post) => post.data.category === activeCategory);

  return (
    <>
      <div className="mb-8 flex flex-wrap gap-2">
        <button
          data-category="all"
          className={`rounded-full border px-4 py-1.5 text-sm font-medium transition-all ${
            activeCategory === "all"
              ? "border-accent bg-accent text-white"
              : "border-border hover:border-accent hover:text-accent"
          }`}
          onClick={() => setActiveCategory("all")}
        >
          전체
        </button>
        {categories.map((category) => (
          <button
            key={category}
            className={`rounded-full border px-4 py-1.5 text-sm font-medium transition-all ${
              activeCategory === category
                ? "border-accent bg-accent text-white"
                : "border-border hover:border-accent hover:text-accent"
            }`}
            onClick={() => setActiveCategory(category)}
          >
            {category}
          </button>
        ))}
      </div>

      <div className="grid grid-cols-1 gap-8 sm:grid-cols-2 lg:grid-cols-3">
        {filteredPosts.map((post) => (
          <CardBlog key={post.id} post={post} />
        ))}
      </div>
    </>
  );
}
