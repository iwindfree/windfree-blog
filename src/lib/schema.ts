import { z } from "zod";
import { SITE } from "@/config";

export const BLOG_PATH = "src/data/blog";

export const blogSchema = z.object({
  author: z.string().default(SITE.author),
  pubDatetime: z.coerce.date(),
  modDatetime: z.coerce.date().optional().nullable(),
  title: z.string(),
  featured: z.boolean().optional(),
  draft: z.boolean().optional(),
  tags: z.array(z.string()).default(["others"]),
  ogImage: z.string().optional(),
  description: z.string(),
  canonicalURL: z.string().optional(),
  hideEditPost: z.boolean().optional(),
  timezone: z.string().optional(),
  category: z
    .enum([
      "MAUI 기본",
      "MAUI 활용",
      "IT 잡썰",
      "LLM Engineering",
      "AI Agent Engineering",
      "RUST",
      "JAVA BCI",
      "프로그래밍 노트",
    ])
    .optional(),
  series: z.string().optional(),
  seriesOrder: z.number().optional(),
});

export type BlogFrontmatter = z.infer<typeof blogSchema>;
