import type { BlogPost } from "@/lib/posts";

type GroupKey = string | number | symbol;

interface GroupFunction<T> {
  (item: T, index?: number): GroupKey;
}

const getPostsByGroupCondition = (
  posts: BlogPost[],
  groupFunction: GroupFunction<BlogPost>
) => {
  const result: Record<GroupKey, BlogPost[]> = {};
  for (let i = 0; i < posts.length; i++) {
    const item = posts[i];
    const groupKey = groupFunction(item, i);
    if (!result[groupKey]) {
      result[groupKey] = [];
    }
    result[groupKey].push(item);
  }
  return result;
};

export default getPostsByGroupCondition;
