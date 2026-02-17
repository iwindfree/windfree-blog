import { unified } from "unified";
import remarkParse from "remark-parse";
import remarkGfm from "remark-gfm";
import remarkRehype from "remark-rehype";
import rehypeRaw from "rehype-raw";
import rehypeSlug from "rehype-slug";
import rehypeStringify from "rehype-stringify";
import rehypeShiki from "@shikijs/rehype";
import {
  transformerNotationDiff,
  transformerNotationHighlight,
  transformerNotationWordHighlight,
} from "@shikijs/transformers";

export async function renderMarkdown(source: string): Promise<string> {
  const result = await unified()
    .use(remarkParse)
    .use(remarkGfm)
    .use(remarkRehype, { allowDangerousHtml: true })
    .use(rehypeRaw)
    .use(rehypeSlug)
    .use(rehypeShiki as any, {
      themes: { light: "github-light", dark: "github-dark" },
      defaultColor: false,
      wrap: true,
      transformers: [
        transformerNotationHighlight(),
        transformerNotationWordHighlight(),
        transformerNotationDiff({ matchAlgorithm: "v3" }),
      ],
    })
    .use(rehypeStringify, { allowDangerousHtml: true })
    .process(source);

  return String(result);
}
