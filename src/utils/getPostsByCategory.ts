import type { BlogPost } from "@/lib/posts";
import getSortedPosts from "./getSortedPosts";

const getPostsByCategory = (posts: BlogPost[], category: string) =>
  getSortedPosts(posts).filter((post) => post.data.category === category);

export default getPostsByCategory;
