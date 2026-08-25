"""Split sections into the chunks that get embedded.

    data/processed/sections.jsonl  ->  data/processed/chunks.jsonl

Sections run from 21 to 13,901 chars. Long ones split at h3 headings, and
between paragraphs when a section has none. Each chunk leads with a breadcrumb
naming its Law, clause and subsection, because subsection titles repeat:
"Procedure" appears three times in the corpus and "Advantage" twice.
"""

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path

PROCESSED = Path("data/processed")
MAX_CHARS = 2000
SLUG_CHARS = 40

H3 = re.compile(r"^### (.+)$", re.MULTILINE)


@dataclass
class Chunk:
    id: str
    section_id: str
    law_number: int
    law_title: str
    clause: str | None
    clause_title: str
    group: str | None
    subsection: str | None
    breadcrumb: str
    body: str
    url: str


def embed_text(chunk: Chunk) -> str:
    """The breadcrumb, then the body."""
    return f"{chunk.breadcrumb}\n\n{chunk.body}"


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")[:SLUG_CHARS]


def _blocks(markdown: str) -> list[tuple[str | None, str]]:
    """Split at h3 headings, keeping any text before the first one."""
    parts = H3.split(markdown)
    blocks = []
    if preamble := parts[0].strip():
        blocks.append((None, preamble))
    for heading, body in zip(parts[1::2], parts[2::2]):
        blocks.append((heading.strip(), body.strip()))
    return blocks


def _pack(body: str, limit: int) -> list[str]:
    """Group paragraphs into runs under the limit, never splitting one."""
    packed: list[str] = []
    current: list[str] = []
    size = 0

    for paragraph in (p for p in body.split("\n\n") if p.strip()):
        if current and size + len(paragraph) > limit:
            packed.append("\n\n".join(current))
            current, size = [], 0
        current.append(paragraph)
        size += len(paragraph) + 2

    if current:
        packed.append("\n\n".join(current))
    return packed


def _trail(section: dict) -> list[str]:
    trail = [f"Law {section['law_number']} {section['law_title']}"]
    if clause := section["clause"]:
        trail.append(f"{clause} {section['clause_title']}")
    elif section["clause_title"]:
        trail.append(section["clause_title"])
    return trail


def _unique(name: str, taken: set[str]) -> str:
    """_slug truncates, so two long titles in one clause can produce one name."""
    base, suffix = name, 2
    while name in taken:
        name = f"{base}-{suffix}"
        suffix += 1
    taken.add(name)
    return name


def chunk_section(section: dict) -> list[Chunk]:
    trail = _trail(section)
    chunks: list[Chunk] = []
    group: str | None = None
    taken: set[str] = set()

    for heading, body in _blocks(section["markdown"]):
        # An all-caps h3 labels the group its siblings belong to, not a rule of
        # its own. Law 12.4 files "Sending-off offences" under PLAYERS and
        # "Sending-off" under TEAM OFFICIALS, and only the group tells them apart.
        if heading and heading.isupper():
            group, heading = heading, None
        if not body:
            continue

        breadcrumb = " > ".join(trail + [c for c in (group, heading) if c])
        name = _unique(_slug(heading or group or "intro"), taken)
        parts = _pack(body, MAX_CHARS)

        for index, part in enumerate(parts, start=1):
            chunks.append(
                Chunk(
                    id=f"{section['id']}#{name}" + (f"-p{index}" if len(parts) > 1 else ""),
                    section_id=section["id"],
                    law_number=section["law_number"],
                    law_title=section["law_title"],
                    clause=section["clause"],
                    clause_title=section["clause_title"],
                    group=group,
                    subsection=heading,
                    breadcrumb=breadcrumb,
                    body=part,
                    url=section["url"],
                )
            )

    return chunks


def main() -> None:
    path = PROCESSED / "sections.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"{path} missing, run `make parse` first")

    sections = [json.loads(line) for line in path.open(encoding="utf-8")]
    chunks = [c for s in sections for c in chunk_section(s)]

    out = PROCESSED / "chunks.jsonl"
    with out.open("w", encoding="utf-8") as handle:
        for chunk in chunks:
            handle.write(json.dumps(asdict(chunk), ensure_ascii=False) + "\n")

    sizes = sorted(len(embed_text(c)) for c in chunks)
    print(
        f"{len(sections)} sections -> {len(chunks)} chunks"
        f"\nembed size  min {sizes[0]}  median {sizes[len(sizes) // 2]}  max {sizes[-1]}"
        f"\nwritten to {out}"
    )
    for chunk in chunks:
        if len(chunk.body) > MAX_CHARS:
            print(f"  over {MAX_CHARS}: {chunk.id} ({len(chunk.body)})")


if __name__ == "__main__":
    main()
