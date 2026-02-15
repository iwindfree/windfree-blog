#!/usr/bin/env python3
"""
WordPress XML to Markdown migration script.
Converts WordPress posts to Astro-compatible Markdown files.
"""

import os
import re
import json
import shutil
import html
from pathlib import Path
from urllib.parse import urlparse, unquote
from lxml import etree

# === Configuration ===
WP_XML = "/Users/windfree/Workspace/blog_bak/WordPress.2026-02-15.xml"
WP_UPLOADS = "/Users/windfree/Workspace/blog_bak/iwindfree/www/wp-content/uploads"
OUTPUT_DIR = "/Users/windfree/Workspace/ws.blog/src/data/blog"
IMG_OUTPUT_DIR = "/Users/windfree/Workspace/ws.blog/public/images/blog"

NS = {
    "wp": "http://wordpress.org/export/1.2/",
    "content": "http://purl.org/rss/1.0/modules/content/",
    "dc": "http://purl.org/dc/elements/1.1/",
    "excerpt": "http://wordpress.org/export/1.2/excerpt/",
}

CATEGORY_MAP = {
    "MAUI 기본": "MAUI 기본",
    "MAUI 활용": "MAUI 활용",
    "IT 잡썰": "IT 잡썰",
    "AI Engineering": "AI Engineering",
    "RUST": "RUST",
}

BOOKSTORE_PATTERN = re.compile(r"BookStore.*\((\d+)\)")


def decode_html(text):
    """Decode HTML entities."""
    return html.unescape(text)


def convert_inline_html(text):
    """Convert inline HTML to markdown."""
    text = re.sub(r'<a[^>]+href="([^"]+)"[^>]*>(.*?)</a>', r'[\2](\1)', text)
    text = re.sub(r'<(?:strong|b)>(.*?)</(?:strong|b)>', r'**\1**', text, flags=re.DOTALL)
    text = re.sub(r'<(?:em|i)>(.*?)</(?:em|i)>', r'*\1*', text, flags=re.DOTALL)
    text = re.sub(r'<code>(.*?)</code>', r'`\1`', text, flags=re.DOTALL)
    text = re.sub(r'<br\s*/?>', '\n', text)
    text = re.sub(r'<[^>]+>', '', text)
    text = decode_html(text)
    return text.strip()


def url_to_local_path(url):
    """Convert WordPress image URL to local blog path."""
    parsed = urlparse(url)
    path = unquote(parsed.path)
    uploads_idx = path.find("/wp-content/uploads/")
    if uploads_idx >= 0:
        relative = path[uploads_idx + len("/wp-content/uploads/"):]
        return f"/images/blog/{relative}"
    return url


def parse_blocks(content):
    """Parse WordPress block content into a list of blocks."""
    if not content:
        return []

    blocks = []
    # Match both self-closing and paired blocks
    # Self-closing: <!-- wp:type {"attr":...} /-->
    # Paired: <!-- wp:type {"attr":...} --> content <!-- /wp:type -->
    pos = 0
    open_pattern = re.compile(r'<!-- wp:(\S+?)(?:\s+(\{.*?\}))?\s*(/)?-->')

    while pos < len(content):
        m = open_pattern.search(content, pos)
        if not m:
            # Remaining text
            remaining = content[pos:].strip()
            if remaining:
                blocks.append({"type": "__text__", "html": remaining})
            break

        # Text before this block
        before = content[pos:m.start()].strip()
        if before:
            blocks.append({"type": "__text__", "html": before})

        block_type = m.group(1)
        attrs_str = m.group(2) or ""
        self_closing = m.group(3) == "/"

        if self_closing:
            blocks.append({"type": block_type, "attrs": attrs_str, "html": ""})
            pos = m.end()
        else:
            # Find closing tag
            close_tag = f"<!-- /wp:{block_type} -->"
            close_idx = content.find(close_tag, m.end())
            if close_idx >= 0:
                inner = content[m.end():close_idx]
                blocks.append({"type": block_type, "attrs": attrs_str, "html": inner})
                pos = close_idx + len(close_tag)
            else:
                # No closing tag found, treat rest as content
                blocks.append({"type": block_type, "attrs": attrs_str, "html": content[m.end():]})
                break

    return blocks


def convert_block(block, images_used):
    """Convert a single block to markdown."""
    btype = block["type"]
    attrs_str = block.get("attrs", "")
    bhtml = block.get("html", "")

    attrs = {}
    if attrs_str:
        try:
            attrs = json.loads(attrs_str)
        except json.JSONDecodeError:
            pass

    if btype == "__text__":
        text = convert_inline_html(bhtml)
        return f"\n{text}\n" if text else ""

    elif btype == "paragraph":
        text = convert_inline_html(bhtml)
        return f"\n{text}\n" if text else ""

    elif btype == "heading":
        level = attrs.get("level", 2)
        # Also try from HTML tag
        if not attrs:
            lm = re.search(r'<h(\d)', bhtml)
            if lm:
                level = int(lm.group(1))
        text = re.sub(r'<h[1-6][^>]*>(.*?)</h[1-6]>', r'\1', bhtml, flags=re.DOTALL)
        text = convert_inline_html(text)
        return f"\n{'#' * level} {text}\n" if text else ""

    elif btype == "code":
        code_match = re.search(r'<code[^>]*>(.*?)</code>', bhtml, re.DOTALL)
        if code_match:
            code = code_match.group(1)
            code = re.sub(r'<[^>]+>', '', code)
            code = decode_html(code)
            lang_match = re.search(r'class="[^"]*language-(\w+)', bhtml)
            lang = lang_match.group(1) if lang_match else ""
            return f"\n```{lang}\n{code}\n```\n"
        return ""

    elif btype == "kevinbatdorf/code-block-pro":
        code = attrs.get("code", "")
        lang = attrs.get("language", "")
        code = decode_html(code)
        return f"\n```{lang}\n{code}\n```\n" if code else ""

    elif btype in ("image", "uagb/image"):
        url = attrs.get("url", "")
        if not url:
            img_match = re.search(r'<img[^>]+src="([^"]+)"', bhtml)
            if img_match:
                url = img_match.group(1)
        if not url:
            return ""

        alt_match = re.search(r'alt="([^"]*)"', bhtml)
        alt = alt_match.group(1) if alt_match else ""

        caption_match = re.search(r'<figcaption[^>]*>(.*?)</figcaption>', bhtml, re.DOTALL)
        caption = ""
        if caption_match:
            caption = re.sub(r'<[^>]+>', '', caption_match.group(1)).strip()

        local_path = url_to_local_path(url)
        images_used.add(url)

        result = f"\n![{alt}]({local_path})\n"
        if caption:
            result += f"*{caption}*\n"
        return result

    elif btype == "list":
        is_ordered = '<ol' in bhtml
        items = re.findall(r'<li[^>]*>(.*?)</li>', bhtml, re.DOTALL)
        result = "\n"
        for i, item in enumerate(items):
            text = convert_inline_html(item.strip())
            prefix = f"{i+1}." if is_ordered else "-"
            result += f"{prefix} {text}\n"
        return result + "\n"

    elif btype == "quote":
        text = re.sub(r'<[^>]+>', '', bhtml)
        text = decode_html(text).strip()
        if text:
            return "\n" + "\n".join(f"> {l}" for l in text.split("\n")) + "\n"
        return ""

    elif btype == "separator":
        return "\n---\n"

    elif btype == "table":
        rows = re.findall(r'<tr[^>]*>(.*?)</tr>', bhtml, re.DOTALL)
        if not rows:
            return ""
        md_rows = []
        for i, row in enumerate(rows):
            cells = re.findall(r'<(?:td|th)[^>]*>(.*?)</(?:td|th)>', row, re.DOTALL)
            cells_text = [convert_inline_html(c) for c in cells]
            md_rows.append("| " + " | ".join(cells_text) + " |")
            if i == 0:
                md_rows.append("| " + " | ".join(["---"] * len(cells_text)) + " |")
        return "\n" + "\n".join(md_rows) + "\n"

    elif btype in ("group", "columns", "column", "uagb/container"):
        inner_blocks = parse_blocks(bhtml)
        return "".join(convert_block(b, images_used) for b in inner_blocks)

    elif btype == "html":
        clean = bhtml.strip()
        return f"\n{clean}\n" if clean else ""

    elif btype == "list-item":
        return ""  # handled by parent list block

    else:
        # Unknown block: try to recursively parse for inner blocks
        inner_blocks = parse_blocks(bhtml)
        if any(b["type"] != "__text__" for b in inner_blocks):
            return "".join(convert_block(b, images_used) for b in inner_blocks)
        text = re.sub(r'<[^>]+>', '', bhtml)
        text = decode_html(text).strip()
        return f"\n{text}\n" if text else ""


def wp_content_to_markdown(content, images_used):
    """Convert WordPress block content to Markdown."""
    blocks = parse_blocks(content)
    parts = [convert_block(b, images_used) for b in blocks]
    md = "".join(parts)
    md = re.sub(r'\n{3,}', '\n\n', md)
    return md.strip()


def copy_image(src_url, wp_uploads_dir, img_output_dir):
    """Copy image from WP uploads to blog public directory."""
    parsed = urlparse(src_url)
    path = unquote(parsed.path)
    uploads_idx = path.find("/wp-content/uploads/")
    if uploads_idx < 0:
        return False

    relative = path[uploads_idx + len("/wp-content/uploads/"):]
    src_path = os.path.join(wp_uploads_dir, relative)
    dst_path = os.path.join(img_output_dir, relative)

    if not os.path.exists(src_path):
        print(f"  WARNING: Image not found: {src_path}")
        return False

    os.makedirs(os.path.dirname(dst_path), exist_ok=True)
    shutil.copy2(src_path, dst_path)
    return True


def generate_slug(title, wp_slug):
    """Generate a clean slug from WP slug or title."""
    if wp_slug and not wp_slug.startswith("%"):
        return wp_slug
    slug = title.lower()
    slug = re.sub(r'\[.*?\]\s*', '', slug)
    slug = slug.replace(" ", "-")
    slug = re.sub(r'[^a-z0-9\-]', '', slug)
    slug = re.sub(r'-+', '-', slug).strip('-')
    return slug or wp_slug


def detect_series(title):
    """Detect if post belongs to a series."""
    match = BOOKSTORE_PATTERN.search(title)
    if match:
        return "MAUI BookStore 만들기", int(match.group(1))
    return None, None


def main():
    print("=== WordPress to Markdown Migration ===\n")

    parser = etree.XMLParser(recover=True)
    tree = etree.parse(WP_XML, parser)
    root = tree.getroot()

    items = root.findall(".//channel/item")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(IMG_OUTPUT_DIR, exist_ok=True)

    post_count = 0
    image_count = 0
    all_images = set()

    for item in items:
        post_type = item.find("wp:post_type", NS)
        status = item.find("wp:status", NS)

        if post_type is None or post_type.text != "post":
            continue
        if status is None or status.text != "publish":
            continue

        title = item.find("title").text or ""
        content_el = item.find("content:encoded", NS)
        content = content_el.text if content_el is not None else ""
        slug = item.find("wp:post_name", NS).text or ""
        date_gmt = item.find("wp:post_date_gmt", NS).text or ""
        excerpt_el = item.find("excerpt:encoded", NS)
        excerpt = excerpt_el.text if excerpt_el is not None else ""

        categories = [c.text for c in item.findall('category[@domain="category"]')]
        tags = [c.text for c in item.findall('category[@domain="post_tag"]')]

        category = None
        for cat in categories:
            if cat in CATEGORY_MAP:
                category = CATEGORY_MAP[cat]
                break

        clean_slug = generate_slug(title, slug)
        series, series_order = detect_series(title)

        images_used = set()
        markdown_content = wp_content_to_markdown(content, images_used)
        all_images.update(images_used)

        # Generate description
        description = ""
        if excerpt:
            description = re.sub(r'<[^>]+>', '', excerpt).strip()
        if not description:
            plain = re.sub(r'[#*\[\]()!`>]', '', markdown_content)
            plain = re.sub(r'\s+', ' ', plain).strip()
            description = plain[:150].rsplit(' ', 1)[0] if len(plain) > 150 else plain

        # Escape quotes in description
        description = description.replace('"', '\\"')

        # Build frontmatter
        fm = f'---\ntitle: "{title}"\n'
        fm += f"author: iwindfree\n"
        fm += f"pubDatetime: {date_gmt.replace(' ', 'T')}Z\n"
        fm += f'slug: "{clean_slug}"\n'
        if category:
            fm += f'category: "{category}"\n'
        if tags:
            fm += f"tags: [{', '.join(json.dumps(t) for t in tags)}]\n"
        else:
            fm += 'tags: ["others"]\n'
        fm += f'description: "{description}"\n'
        if series:
            fm += f'series: "{series}"\n'
            fm += f"seriesOrder: {series_order}\n"
        fm += "---\n"

        filename = f"{clean_slug}.md"
        filepath = os.path.join(OUTPUT_DIR, filename)
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(fm)
            f.write("\n")
            f.write(markdown_content)
            f.write("\n")

        post_count += 1
        print(f"  [{post_count}] {title} -> {filename} ({len(images_used)} images)")

    print(f"\n--- Copying {len(all_images)} images ---")
    for src_url in all_images:
        if copy_image(src_url, WP_UPLOADS, IMG_OUTPUT_DIR):
            image_count += 1

    print(f"\n=== Done! ===")
    print(f"  Posts: {post_count}")
    print(f"  Images copied: {image_count}")


if __name__ == "__main__":
    main()
