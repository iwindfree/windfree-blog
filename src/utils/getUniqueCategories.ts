import type { CollectionEntry } from "astro:content";
import postFilter from "./postFilter";

const getUniqueCategories = (posts: CollectionEntry<"blog">[]) => {
  const categories = posts
    .filter(postFilter)
    .map(post => post.data.category)
    .filter((category): category is NonNullable<typeof category> => !!category);

  const uniqueCategories = [...new Set(categories)].sort();
  return uniqueCategories;
};

export default getUniqueCategories;
