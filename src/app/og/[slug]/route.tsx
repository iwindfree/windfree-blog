import { ImageResponse } from "@vercel/og";
import { getPostBySlug, getAllPosts } from "@/lib/posts";
import { SITE } from "@/config";

export const runtime = "nodejs";

export async function generateStaticParams() {
  if (!SITE.dynamicOgImage) return [];
  const posts = getAllPosts().filter((p) => !p.data.draft && !p.data.ogImage);
  return posts.map((post) => ({ slug: post.slug }));
}

export async function GET(
  _request: Request,
  { params }: { params: Promise<{ slug: string }> }
) {
  const { slug } = await params;

  // Handle site-level OG image
  if (slug === "site") {
    return new ImageResponse(
      (
        <div
          style={{
            background: "#FAFBFD",
            width: "100%",
            height: "100%",
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
          }}
        >
          <div
            style={{
              position: "absolute",
              top: "-1px",
              right: "-1px",
              border: "4px solid #1B3A5C",
              background: "#F0F2F5",
              opacity: 0.9,
              borderRadius: "4px",
              display: "flex",
              justifyContent: "center",
              margin: "2.5rem",
              width: "88%",
              height: "80%",
            }}
          />
          <div
            style={{
              border: "4px solid #1B3A5C",
              background: "#FAFBFD",
              borderRadius: "4px",
              display: "flex",
              justifyContent: "center",
              margin: "2rem",
              width: "88%",
              height: "80%",
            }}
          >
            <div
              style={{
                display: "flex",
                flexDirection: "column",
                justifyContent: "space-between",
                margin: "20px",
                width: "90%",
                height: "90%",
              }}
            >
              <div
                style={{
                  display: "flex",
                  flexDirection: "column",
                  justifyContent: "center",
                  alignItems: "center",
                  height: "90%",
                  maxHeight: "90%",
                  overflow: "hidden",
                  textAlign: "center",
                }}
              >
                <p style={{ fontSize: 72, fontWeight: "bold" }}>
                  {SITE.title}
                </p>
                <p style={{ fontSize: 28 }}>{SITE.desc}</p>
              </div>
              <div
                style={{
                  display: "flex",
                  justifyContent: "flex-end",
                  width: "100%",
                  marginBottom: "8px",
                  fontSize: 28,
                }}
              >
                <span style={{ overflow: "hidden", fontWeight: "bold" }}>
                  {new URL(SITE.website).hostname}
                </span>
              </div>
            </div>
          </div>
        </div>
      ),
      { width: 1200, height: 630 }
    );
  }

  const post = getPostBySlug(slug);

  if (!post || !SITE.dynamicOgImage) {
    return new Response("Not found", { status: 404 });
  }

  return new ImageResponse(
    (
      <div
        style={{
          background: "#FAFBFD",
          width: "100%",
          height: "100%",
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
        }}
      >
        <div
          style={{
            position: "absolute",
            top: "-1px",
            right: "-1px",
            border: "4px solid #1B3A5C",
            background: "#F0F2F5",
            opacity: 0.9,
            borderRadius: "4px",
            display: "flex",
            justifyContent: "center",
            margin: "2.5rem",
            width: "88%",
            height: "80%",
          }}
        />
        <div
          style={{
            border: "4px solid #1B3A5C",
            background: "#FAFBFD",
            borderRadius: "4px",
            display: "flex",
            justifyContent: "center",
            margin: "2rem",
            width: "88%",
            height: "80%",
          }}
        >
          <div
            style={{
              display: "flex",
              flexDirection: "column",
              justifyContent: "space-between",
              margin: "20px",
              width: "90%",
              height: "90%",
            }}
          >
            <p
              style={{
                fontSize: 72,
                fontWeight: "bold",
                maxHeight: "84%",
                overflow: "hidden",
              }}
            >
              {post.data.title}
            </p>
            <div
              style={{
                display: "flex",
                justifyContent: "space-between",
                width: "100%",
                marginBottom: "8px",
                fontSize: 28,
              }}
            >
              <span>
                by{" "}
                <span style={{ overflow: "hidden", fontWeight: "bold" }}>
                  {post.data.author}
                </span>
              </span>
              <span style={{ overflow: "hidden", fontWeight: "bold" }}>
                {SITE.title}
              </span>
            </div>
          </div>
        </div>
      </div>
    ),
    { width: 1200, height: 630 }
  );
}
