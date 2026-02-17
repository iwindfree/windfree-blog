import fs from "fs";
import path from "path";
import matter from "gray-matter";
import { blogSchema, BLOG_PATH, type BlogFrontmatter } from "./schema";

export interface BlogPost {
  id: string;
  slug: string;
  data: BlogFrontmatter;
  body: string;
  filePath: string;
}

function getMdFiles(dir: string): string[] {
  const entries = fs.readdirSync(dir, { withFileTypes: true });
  const files: string[] = [];

  for (const entry of entries) {
    const fullPath = path.join(dir, entry.name);
    if (entry.isDirectory()) {
      if (!entry.name.startsWith("_")) {
        files.push(...getMdFiles(fullPath));
      }
    } else if (
      (entry.name.endsWith(".md") || entry.name.endsWith(".mdx")) &&
      !entry.name.startsWith("_")
    ) {
      files.push(fullPath);
    }
  }

  return files;
}

let cachedPosts: BlogPost[] | null = null;

export function getAllPosts(): BlogPost[] {
  if (cachedPosts) return cachedPosts;

  const blogDir = path.join(process.cwd(), BLOG_PATH);
  const files = getMdFiles(blogDir);

  const posts = files
    .map((filePath) => {
      const fileContent = fs.readFileSync(filePath, "utf-8");
      const { data: rawData, content } = matter(fileContent);

      const parsed = blogSchema.safeParse(rawData);
      if (!parsed.success) {
        console.warn(`Invalid frontmatter in ${filePath}:`, parsed.error);
        return null;
      }

      const relativePath = path.relative(process.cwd(), filePath);
      const fileName = path.basename(filePath, path.extname(filePath));

      // Use slug from frontmatter if available, otherwise use filename
      const slug = (rawData.slug as string) || fileName;

      return {
        id: slug,
        slug,
        data: parsed.data,
        body: content,
        filePath: relativePath,
      } satisfies BlogPost;
    })
    .filter((post): post is BlogPost => post !== null);

  cachedPosts = posts;
  return posts;
}

export function getPostBySlug(slug: string): BlogPost | undefined {
  return getAllPosts().find((post) => post.slug === slug);
}
