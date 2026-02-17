/** post.body (Astro 5 glob loader raw markdown)에서 첫 마크다운 이미지 URL 추출 */
export function getFirstImage(body: string | undefined): string | null {
  if (!body) return null;
  const match = body.match(/!\[.*?\]\((.*?)\)/);
  return match?.[1] ?? null;
}
