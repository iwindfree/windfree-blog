import { BLOG_PATH } from "@/lib/schema";
import { slugifyStr } from "./slugify";

export function getPath(
  id: string,
  filePath: string | undefined,
  includeBase = true
) {
  const pathSegments = filePath
    ?.replace(BLOG_PATH, "")
    .split("/")
    .filter((path) => path !== "")
    .filter((path) => !path.startsWith("_"))
    .slice(0, -1)
    .map((segment) => slugifyStr(segment));

  const basePath = includeBase ? "/blog" : "";

  const blogId = id.split("/");
  const slug = blogId.length > 0 ? blogId.slice(-1) : blogId;

  if (!pathSegments || pathSegments.length < 1) {
    return [basePath, slug].join("/");
  }

  return [basePath, ...pathSegments, slug].join("/");
}
