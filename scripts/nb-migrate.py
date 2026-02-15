#!/usr/bin/env python3
"""
Jupyter Notebook to Markdown migration script.
Converts notebooks to Astro-compatible Markdown files for blog posts.
"""

import os
import re
import json
import base64
import hashlib
from pathlib import Path
from datetime import datetime
import nbformat

# === Configuration ===
NB_BASE = "/Users/windfree/Workspace/ws.study/ai-engineering"
OUTPUT_DIR = "/Users/windfree/Workspace/ws.blog/src/data/blog"
IMG_OUTPUT_DIR = "/Users/windfree/Workspace/ws.blog/public/images/notebooks"

# Notebook mapping: (relative_path, slug_prefix, pubdate, tags)
# Dates are spread out to create a nice timeline
NOTEBOOKS = [
    # 01 - Tokenization and Inference
    ("llm_engineering/01_tokenization_and_inference/01-1.llm_token_basic.ipynb",
     "llm-token-basic", "2025-01-06T09:00:00Z", ["ai", "llm", "tokenization"]),
    ("llm_engineering/01_tokenization_and_inference/01-2.llm_tokenizer_advanced.ipynb",
     "llm-tokenizer-advanced", "2025-01-07T09:00:00Z", ["ai", "llm", "tokenization"]),
    ("llm_engineering/01_tokenization_and_inference/01-3.llm_inference.ipynb",
     "llm-inference", "2025-01-08T09:00:00Z", ["ai", "llm", "inference"]),
    # 02 - LLM API
    ("llm_engineering/02_llm_api/02-1.llm_api_basic.ipynb",
     "llm-api-basic", "2025-01-13T09:00:00Z", ["ai", "llm", "api"]),
    ("llm_engineering/02_llm_api/02-2.llm_api_intermediate.ipynb",
     "llm-api-intermediate", "2025-01-14T09:00:00Z", ["ai", "llm", "api"]),
    ("llm_engineering/02_llm_api/02-3.llm_api_advanced.ipynb",
     "llm-api-advanced", "2025-01-15T09:00:00Z", ["ai", "llm", "api"]),
    # 03 - Prompting and Multimodal
    ("llm_engineering/03_prompting_and_multimodal/03-1.system_message.ipynb",
     "llm-system-message", "2025-01-20T09:00:00Z", ["ai", "llm", "prompting"]),
    ("llm_engineering/03_prompting_and_multimodal/03-2.[실습]meeting_minutes_summary.ipynb",
     "llm-meeting-minutes-summary", "2025-01-21T09:00:00Z", ["ai", "llm", "prompting"]),
    ("llm_engineering/03_prompting_and_multimodal/03-3.multi_modal_basic.ipynb",
     "llm-multimodal-basic", "2025-01-22T09:00:00Z", ["ai", "llm", "multimodal"]),
    # 04 - Frameworks and Tools
    ("llm_engineering/04_frameworks_and_tools/04-1.using_gradio.ipynb",
     "llm-using-gradio", "2025-01-27T09:00:00Z", ["ai", "llm", "gradio"]),
    ("llm_engineering/04_frameworks_and_tools/04-2.using_huggingface.ipynb",
     "llm-using-huggingface", "2025-01-28T09:00:00Z", ["ai", "llm", "huggingface"]),
    ("llm_engineering/04_frameworks_and_tools/04-3.using_google_colab.ipynb",
     "llm-using-google-colab", "2025-01-29T09:00:00Z", ["ai", "llm", "colab"]),
    # 05 - Tool Use and Benchmarks
    ("llm_engineering/05_tool_use_and_benchmarks/05-1.llm_use_tools.ipynb",
     "llm-use-tools", "2025-02-03T09:00:00Z", ["ai", "llm", "tool-use"]),
    ("llm_engineering/05_tool_use_and_benchmarks/05-2.llm_benchmarks_guide.ipynb",
     "llm-benchmarks-guide", "2025-02-04T09:00:00Z", ["ai", "llm", "benchmarks"]),
    # 06 - Using RAG
    ("llm_engineering/06_using_rag/06-1.vector_embeddings_and_rag.ipynb",
     "llm-vector-embeddings-rag", "2025-02-10T09:00:00Z", ["ai", "llm", "rag", "embeddings"]),
    ("llm_engineering/06_using_rag/06-2.LangChain_vs_LiteLLM.ipynb",
     "llm-langchain-vs-litellm", "2025-02-11T09:00:00Z", ["ai", "llm", "rag", "langchain"]),
    ("llm_engineering/06_using_rag/06-3.advanced_rag_with_vectordb.ipynb",
     "llm-advanced-rag-vectordb", "2025-02-12T09:00:00Z", ["ai", "llm", "rag", "vectordb"]),
    ("llm_engineering/06_using_rag/06-4.chatbot_with_langchain.ipynb",
     "llm-chatbot-langchain", "2025-02-13T09:00:00Z", ["ai", "llm", "rag", "chatbot"]),
    # 07 - RAG Evaluation
    ("llm_engineering/07_rag_evaluation/07-1.rag_evaluation.ipynb",
     "llm-rag-evaluation", "2025-02-17T09:00:00Z", ["ai", "llm", "rag", "evaluation"]),
    ("llm_engineering/07_rag_evaluation/07-2.example_rag_evaluation_advanced.ipynb",
     "llm-rag-evaluation-advanced", "2025-02-18T09:00:00Z", ["ai", "llm", "rag", "evaluation"]),
    # 08 - Fine Tuning
    ("llm_engineering/08_fine_tuning_with_frontier_model/08-1.dataset.ipynb",
     "llm-fine-tuning-dataset", "2025-02-24T09:00:00Z", ["ai", "llm", "fine-tuning"]),
    # Agent Engineering
    ("agent_engineering/01.building_effective_agents.ipynb",
     "building-effective-agents", "2025-03-03T09:00:00Z", ["ai", "llm", "agents"]),
]

# Noise patterns in output to filter
NOISE_PATTERNS = [
    re.compile(r'^(Collecting|Downloading|Installing|Requirement already|Successfully|Using cached|Building|Preparing|Creating|WARNING|Note:|ERROR:)', re.MULTILINE),
    re.compile(r'^\s*━+', re.MULTILINE),
    re.compile(r'^\s*$', re.MULTILINE),
]


def is_noise_output(text):
    """Check if output is mostly noise (pip install, warnings, etc.)."""
    lines = text.strip().split('\n')
    if not lines:
        return True
    noise_lines = sum(1 for l in lines if any(p.match(l) for p in NOISE_PATTERNS))
    return noise_lines > len(lines) * 0.7


def extract_title(nb):
    """Extract title from the first markdown cell's # heading."""
    for cell in nb.cells:
        if cell.cell_type == 'markdown':
            match = re.match(r'^#\s+(.+)', cell.source, re.MULTILINE)
            if match:
                return match.group(1).strip()
    return "Untitled"


def extract_images(cell, slug, img_counter):
    """Extract base64 images from cell outputs and save them."""
    images = []
    for output in cell.get('outputs', []):
        data = output.get('data', {})
        for mime_type in ['image/png', 'image/jpeg', 'image/svg+xml']:
            if mime_type in data:
                img_data = data[mime_type]
                ext = mime_type.split('/')[-1]
                if ext == 'svg+xml':
                    ext = 'svg'

                img_counter[0] += 1
                filename = f"{slug}-img-{img_counter[0]}.{ext}"
                img_path = os.path.join(IMG_OUTPUT_DIR, filename)

                os.makedirs(IMG_OUTPUT_DIR, exist_ok=True)

                if isinstance(img_data, str):
                    # Base64 encoded
                    with open(img_path, 'wb') as f:
                        f.write(base64.b64decode(img_data))
                elif isinstance(img_data, list):
                    content = ''.join(img_data)
                    with open(img_path, 'wb') as f:
                        f.write(base64.b64decode(content))

                images.append(f"/images/notebooks/{filename}")
                break  # Only take the first image format
    return images


def convert_output(output):
    """Convert a cell output to markdown text."""
    output_type = output.get('output_type', '')

    if output_type == 'stream':
        text = output.get('text', '')
        if isinstance(text, list):
            text = ''.join(text)
        return text

    elif output_type in ('execute_result', 'display_data'):
        data = output.get('data', {})
        if 'text/plain' in data:
            text = data['text/plain']
            if isinstance(text, list):
                text = ''.join(text)
            return text
        if 'text/html' in data:
            html = data['text/html']
            if isinstance(html, list):
                html = ''.join(html)
            # Strip HTML tags for simple text output
            text = re.sub(r'<[^>]+>', '', html)
            return text.strip()

    elif output_type == 'error':
        traceback = output.get('traceback', [])
        # ANSI color codes removal
        text = '\n'.join(traceback)
        text = re.sub(r'\x1b\[[0-9;]*m', '', text)
        return text

    return ''


def notebook_to_markdown(nb_path, slug):
    """Convert a Jupyter notebook to markdown content."""
    nb = nbformat.read(nb_path, as_version=4)
    title = extract_title(nb)
    parts = []
    img_counter = [0]
    first_heading_skipped = False

    for cell in nb.cells:
        if cell.cell_type == 'markdown':
            source = cell.source.strip()
            if not first_heading_skipped and source.startswith('# '):
                # Skip the first H1 heading (used as title)
                # But keep the rest of the cell content
                lines = source.split('\n')
                remaining = '\n'.join(lines[1:]).strip()
                first_heading_skipped = True
                if remaining:
                    parts.append(remaining)
                continue
            first_heading_skipped = True
            if source:
                parts.append(source)

        elif cell.cell_type == 'code':
            source = cell.source.strip()
            if not source:
                continue

            # Skip cells that are just pip install or environment setup
            if source.startswith('!pip') or source.startswith('%pip') or source.startswith('!apt'):
                continue

            parts.append(f"\n```python\n{source}\n```\n")

            # Process outputs
            outputs = cell.get('outputs', [])
            if outputs:
                # Extract images first
                images = extract_images(cell, slug, img_counter)
                for img in images:
                    parts.append(f"\n![output]({img})\n")

                # Collect text output
                text_parts = []
                for output in outputs:
                    # Skip if this output has image data (already handled)
                    if output.get('data', {}).get('image/png') or output.get('data', {}).get('image/jpeg'):
                        continue
                    text = convert_output(output)
                    if text and not is_noise_output(text):
                        text_parts.append(text.rstrip())

                if text_parts:
                    combined = '\n'.join(text_parts)
                    # Truncate very long outputs
                    lines = combined.split('\n')
                    if len(lines) > 50:
                        combined = '\n'.join(lines[:50]) + f'\n... (출력 {len(lines) - 50}줄 생략)'
                    parts.append(f'\n<div class="nb-output">\n\n```text\n{combined}\n```\n\n</div>\n')

    return title, '\n\n'.join(parts)


def generate_description(content, title):
    """Generate description from content."""
    # Get first meaningful paragraph
    lines = content.split('\n')
    for line in lines:
        line = line.strip()
        if line and not line.startswith('#') and not line.startswith('```') and not line.startswith('|') and not line.startswith('-') and not line.startswith('<') and len(line) > 20:
            desc = re.sub(r'[*`\[\]()]', '', line)
            if len(desc) > 150:
                desc = desc[:150].rsplit(' ', 1)[0]
            return desc
    return title


def main():
    print("=== Jupyter Notebook to Markdown Migration ===\n")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(IMG_OUTPUT_DIR, exist_ok=True)

    post_count = 0
    img_count = 0

    for nb_rel_path, slug, pubdate, tags in NOTEBOOKS:
        nb_path = os.path.join(NB_BASE, nb_rel_path)

        if not os.path.exists(nb_path):
            print(f"  SKIP: {nb_rel_path} (not found)")
            continue

        title, content = notebook_to_markdown(nb_path, slug)
        description = generate_description(content, title)
        description = description.replace('"', '\\"')

        # Count images
        img_files = [f for f in os.listdir(IMG_OUTPUT_DIR) if f.startswith(slug)] if os.path.exists(IMG_OUTPUT_DIR) else []
        img_count += len(img_files)

        # Build frontmatter
        tag_str = ", ".join(f'"{t}"' for t in tags)
        fm = f'---\ntitle: "{title}"\n'
        fm += f"author: iwindfree\n"
        fm += f"pubDatetime: {pubdate}\n"
        fm += f'slug: "{slug}"\n'
        fm += f'category: "AI Engineering"\n'
        fm += f"tags: [{tag_str}]\n"
        fm += f'description: "{description}"\n'
        fm += "---\n"

        filename = f"{slug}.md"
        filepath = os.path.join(OUTPUT_DIR, filename)
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(fm)
            f.write("\n")
            f.write(content)
            f.write("\n")

        post_count += 1
        print(f"  [{post_count}] {title} -> {filename}")

    print(f"\n=== Done! ===")
    print(f"  Posts: {post_count}")
    print(f"  Images: {img_count}")


if __name__ == "__main__":
    main()
