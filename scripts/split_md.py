#!/usr/bin/env python
"""
Split a markdown knowledge base into atomic topics for local RAG.

Usage:
    python split_md_atomic_topics.py \
        --input fictional_knowledge_base.md \
        --jsonl fictional_kb_atomic_topics.jsonl \
        --shard-dir atomic_topics_md

    # only md
    python split_md_atomic_topics.py \
        --input data/fictional_knowledge_base.md \
        --shard-dir data/atomic_topics_md

- JSONL: suitable for custom loaders / vector stores.
- Shard MD files: directly usable with LlamaIndex SimpleDirectoryReader.
"""

import argparse
import json
import re
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Dict, Any, Optional


# ---------- Data model ----------

@dataclass
class AtomicTopic:
    id: str
    path: List[str]          # heading hierarchy
    title: str               # short label
    type: str                # "heading_intro" | "bullet" | "numbered"
    text: str                # atomic chunk content
    start_line: int
    end_line: int
    source: str              # original md file name


# ---------- Helpers ----------

def slugify(s: str) -> str:
    s = re.sub(r"[^\w\s\-]+", "", s, flags=re.UNICODE).strip().lower()
    s = re.sub(r"\s+", "-", s)
    s = re.sub(r"-{2,}", "-", s)
    return s or "untitled"


def current_path_titles(stack) -> List[str]:
    # stack: list[(level, title, start_line)]
    return [t[1] for t in stack]


def add_topic(
    topics: List[AtomicTopic],
    path: List[str],
    title: str,
    typ: str,
    content: str,
    start_line: int,
    end_line: int,
    source: str,
) -> None:
    content = (content or "").strip()
    if not content:
        return

    # Build a reasonably stable id using path + title
    path_slug = ".".join(slugify(p) for p in path) if path else ""
    title_slug = slugify(title) if title else "topic"
    base = ".".join(x for x in [path_slug, title_slug] if x)
    topic_id = base or f"topic-{len(topics)+1}"

    topics.append(
        AtomicTopic(
            id=topic_id,
            path=list(path),
            title=title,
            type=typ,
            text=content,
            start_line=start_line,
            end_line=end_line,
            source=source,
        )
    )


# ---------- Core splitter ----------

def split_markdown_to_atomic_topics(md_path: Path) -> List[AtomicTopic]:
    text = md_path.read_text(encoding="utf-8")
    lines = text.splitlines()
    n = len(lines)

    topics: List[AtomicTopic] = []
    heading_stack: List = []  # list[(level, title, start_line)]

    i = 0
    while i < n:
        line = lines[i]

        # --- Headings (#, ##, ###, ...) ---
        m = re.match(r"^(#{1,6})\s+(.*)$", line)
        if m:
            level = len(m.group(1))
            title = m.group(2).strip()

            # maintain heading stack (like HTML outline)
            while heading_stack and heading_stack[-1][0] >= level:
                heading_stack.pop()
            heading_stack.append((level, title, i))

            # capture an intro paragraph immediately after heading
            j = i + 1
            paras: List[str] = []

            # skip blank lines
            while j < n and lines[j].strip() == "":
                j += 1

            # collect until:
            # - next heading
            # - or list/numbered item (we treat those as separate atomic topics)
            while (
                j < n
                and not re.match(r"^(#{1,6})\s+", lines[j])
                and not re.match(r"^\s*([*\-]|\d+\.)\s+", lines[j])
            ):
                if lines[j].strip():
                    paras.append(lines[j].strip())
                j += 1

            if paras:
                add_topic(
                    topics=topics,
                    path=current_path_titles(heading_stack),
                    title=f"{title} — Overview",
                    typ="heading_intro",
                    content=" ".join(paras),
                    start_line=i,
                    end_line=j - 1,
                    source=md_path.name,
                )

            i += 1
            continue

        # --- Bulleted list item (* / -) ---
        m_bullet = re.match(r"^\s*([*\-])\s+(.*)$", line)
        if m_bullet:
            item = m_bullet.group(2).strip()
            j = i + 1
            cont: List[str] = []

            # capture indented continuation lines until next heading/list
            while (
                j < n
                and not re.match(r"^(#{1,6})\s+", lines[j])
                and not re.match(r"^\s*([*\-]|\d+\.)\s+", lines[j])
                and (lines[j].strip() == "" or re.match(r"^\s{2,}\S", lines[j]))
            ):
                if lines[j].strip():
                    cont.append(lines[j].strip())
                j += 1

            full = item + (" " + " ".join(cont) if cont else "")

            # If bullet starts with **Title**: treat that as the topic title
            title_match = re.match(r"^\*\*(.+?)\*\*:?(\s*.*)$", item)
            if title_match:
                title = title_match.group(1).strip()
                rest = (title_match.group(2) or "").strip()
                content = (rest + (" " + " ".join(cont) if cont else "")).strip() or title
            else:
                # fallback: use text before ':' as a short label
                title = item.split(":", 1)[0].strip()[:80]
                content = full

            add_topic(
                topics=topics,
                path=current_path_titles(heading_stack),
                title=title,
                typ="bullet",
                content=content,
                start_line=i,
                end_line=j - 1,
                source=md_path.name,
            )

            i = j
            continue

        # --- Numbered list item (1. Step) ---
        m_num = re.match(r"^\s*(\d+)\.\s+(.*)$", line)
        if m_num:
            num = m_num.group(1)
            item = m_num.group(2).strip()
            j = i + 1
            cont = []

            while (
                j < n
                and not re.match(r"^(#{1,6})\s+", lines[j])
                and not re.match(r"^\s*([*\-]|\d+\.)\s+", lines[j])
                and (lines[j].strip() == "" or re.match(r"^\s{2,}\S", lines[j]))
            ):
                if lines[j].strip():
                    cont.append(lines[j].strip())
                j += 1

            full = item + (" " + " ".join(cont) if cont else "")
            title = f"{num}. {item.split(':', 1)[0].strip()[:80]}"

            add_topic(
                topics=topics,
                path=current_path_titles(heading_stack),
                title=title,
                typ="numbered",
                content=full,
                start_line=i,
                end_line=j - 1,
                source=md_path.name,
            )

            i = j
            continue

        i += 1

    # sort by original order
    topics.sort(key=lambda t: (t.start_line, t.end_line))
    return topics


# ---------- Output helpers ----------

def save_jsonl(topics: List[AtomicTopic], jsonl_path: Path) -> None:
    with jsonl_path.open("w", encoding="utf-8") as f:
        for t in topics:
            f.write(json.dumps(asdict(t), ensure_ascii=False) + "\n")


def save_md_shards(topics: List[AtomicTopic], shard_dir: Path) -> None:
    shard_dir.mkdir(parents=True, exist_ok=True)

    readme_lines = ["# Atomic Topics (Markdown shards)", ""]
    for t in topics:
        # simple human-readable md per topic
        lines = [
            f"# {t.title}",
            "",
        ]
        if t.path:
            lines.append("**Path:** " + " / ".join(t.path))
        lines.append(f"**Type:** `{t.type}`  ")
        lines.append(f"**Source:** `{t.source}`  ")
        lines.append(f"**Lines:** {t.start_line}–{t.end_line}")
        lines.append("\n---\n")
        lines.append(t.text.strip())
        md_content = "\n".join(lines).strip() + "\n"

        fname = slugify(t.id)[:120] + ".md"
        (shard_dir / fname).write_text(md_content, encoding="utf-8")

        readme_lines.append(f"- [{t.title}]({fname})")

    (shard_dir / "README.md").write_text("\n".join(readme_lines) + "\n", encoding="utf-8")


# ---------- CLI ----------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, required=True, help="Input markdown file")
    ap.add_argument("--jsonl", type=Path, required=False, help="Output JSONL path")
    ap.add_argument("--shard-dir", type=Path, required=False, help="Output dir for atomic MD shards")
    args = ap.parse_args()

    topics = split_markdown_to_atomic_topics(args.input)
    print(f"Extracted {len(topics)} atomic topics from {args.input.name}")

    if args.jsonl:
        save_jsonl(topics, args.jsonl)
        print(f"Wrote JSONL to {args.jsonl}")

    if args.shard_dir:
        save_md_shards(topics, args.shard_dir)
        print(f"Wrote markdown shards to {args.shard_dir}")

    if not args.jsonl and not args.shard_dir:
        # fallback: preview first few topics to stdout
        for t in topics[:10]:
            print(f"- [{t.type}] {t.title} ({t.start_line}–{t.end_line})")


if __name__ == "__main__":
    main()
