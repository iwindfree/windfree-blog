export function getFirstImage(body: string | undefined): string | null {
  if (!body) return null;
  const match = body.match(/!\[.*?\]\((.*?)\)/);
  return match?.[1] ?? null;
}
