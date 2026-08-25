"""Parse the raw Laws of the Game HTML into a corpus and an eval set.

    data/processed/sections.jsonl   the Laws        (indexed)
    data/processed/faqs.jsonl       IFAB's own Q&A  (held out)

The two must never mix. Index the Q&A and a question retrieves its own answer,
which pins every retrieval metric to the ceiling and measures nothing.
"""

import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path

from bs4 import BeautifulSoup, NavigableString, Tag

from lotg.sources import LAW_PAGES, LawPage

RAW_DIR = Path("data/raw/laws")
OUT_DIR = Path("data/processed")

DROP_TAGS = ["script", "style", "svg", "noscript"]
# Wording cut from this edition. Indexing it would state repealed rules as current law.
DELETED_TAGS = ["deletedexplanation", "deleted"]
ADDED_TAGS = ["addedexplanation", "added"]
HEADINGS = {"h1", "h2", "h3", "h4", "h5", "h6"}

# Clause numbers render as separate spans: "12 . 4 Disciplinary action".
CLAUSE_SPACING = re.compile(r"(\d+)\s*\.\s*(\d+)")
CLAUSE_PREFIX = re.compile(r"^(\d+\.\d+)\s*(.*)$")


@dataclass
class Section:
    id: str
    law_number: int
    law_title: str
    clause: str | None
    clause_title: str
    url: str
    markdown: str
    amended: bool
    removed_text: list[str] = field(default_factory=list)
    amendment_notes: list[str] = field(default_factory=list)


@dataclass
class Faq:
    id: str
    law_number: int
    law_title: str
    url: str
    question: str
    answer: str


def _text(node: Tag | NavigableString) -> str:
    raw = node.get_text(" ") if isinstance(node, Tag) else str(node)
    return re.sub(r"\s+", " ", raw).strip()


def _heading(node: Tag) -> str:
    return CLAUSE_SPACING.sub(r"\1.\2", _text(node))


def _prune(root: Tag) -> None:
    for tag in root.find_all(DROP_TAGS + DELETED_TAGS):
        tag.decompose()
    # Clause headings sit inside the collapse toggle, so buttons unwrap, never drop.
    for tag in root.find_all("button"):
        tag.unwrap()


def _render(node: Tag, out: list[str]) -> None:
    for child in node.children:
        if not isinstance(child, Tag):
            continue

        if child.name in HEADINGS:
            if text := _heading(child):
                out.append(f"{'#' * int(child.name[1])} {text}")
        elif child.name == "p":
            if text := _text(child):
                out.append(text)
        elif child.name in {"ul", "ol"}:
            bullet = "1." if child.name == "ol" else "-"
            for item in child.find_all("li", recursive=False):
                if text := _text(item):
                    out.append(f"{bullet} {text}")
        elif child.name == "table":
            for row in child.find_all("tr"):
                cells = [_text(c) for c in row.find_all(["th", "td"])]
                if any(cells):
                    out.append("| " + " | ".join(cells) + " |")
        else:
            _render(child, out)


def _split_clause(heading: str) -> tuple[str | None, str]:
    if match := CLAUSE_PREFIX.match(heading):
        return match.group(1), match.group(2).strip()
    return None, heading


def _parse_sections(soup: BeautifulSoup, page: LawPage) -> list[Section]:
    sections = []

    for index, article in enumerate(soup.find("main").find_all("article")):
        # Read the amendment markup before _prune throws the deleted wording away.
        removed = [t for tag in article.find_all(DELETED_TAGS) if (t := _text(tag))]
        added = bool(article.find_all(ADDED_TAGS))
        notes = [
            note
            for tag in article.find_all("explanation")
            if (note := (tag.get("data-explanation") or "").strip())
        ]

        _prune(article)
        heading = article.find(["h2", "h3"])
        clause, clause_title = _split_clause(_heading(heading) if heading else f"Section {index}")

        lines: list[str] = []
        _render(article, lines)
        markdown = "\n\n".join(lines).strip()
        if not markdown:
            continue

        sections.append(
            Section(
                id=f"law-{page.number:02d}-{clause or f's{index}'}",
                law_number=page.number,
                law_title=page.title,
                clause=clause,
                clause_title=clause_title,
                url=page.url,
                markdown=markdown,
                amended=added or bool(removed),
                removed_text=removed,
                amendment_notes=notes,
            )
        )

    return sections


def _parse_faqs(soup: BeautifulSoup, page: LawPage) -> list[Faq]:
    faqs = []

    for index, container in enumerate(soup.select('[class*="QnAContainer"]'), start=1):
        question_el = container.select_one('[class*="StyledQuestion"]')
        answer_el = container.select_one('[class*="StyledAnswer"]')
        if not question_el or not answer_el:
            continue

        _prune(answer_el)
        question, answer = _text(question_el), _text(answer_el)
        if not question or not answer:
            continue

        faqs.append(
            Faq(
                id=f"faq-{page.number:02d}-{index:04d}",
                law_number=page.number,
                law_title=page.title,
                url=page.url,
                question=question,
                answer=answer,
            )
        )

    return faqs


def _write_jsonl(path: Path, rows: list[Section] | list[Faq]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(asdict(row), ensure_ascii=False) + "\n")


def parse_all() -> tuple[list[Section], list[Faq]]:
    sections: list[Section] = []
    faqs: list[Faq] = []

    for page in LAW_PAGES:
        path = RAW_DIR / f"law-{page.number:02d}-{page.slug}.html"
        if not path.exists():
            raise FileNotFoundError(f"{path} missing, run `make fetch` first")

        soup = BeautifulSoup(path.read_text(encoding="utf-8"), "lxml")
        # FAQs first: _parse_sections mutates the tree.
        page_faqs = _parse_faqs(soup, page)
        page_sections = _parse_sections(soup, page)
        faqs += page_faqs
        sections += page_sections

        print(
            f"Law {page.number:2d}  {page.title:<36} "
            f"{len(page_sections):2d} sections  {len(page_faqs):4d} FAQs"
        )

    return sections, faqs


def main() -> None:
    sections, faqs = parse_all()
    _write_jsonl(OUT_DIR / "sections.jsonl", sections)
    _write_jsonl(OUT_DIR / "faqs.jsonl", faqs)

    chars = sum(len(s.markdown) for s in sections)
    amended = sum(1 for s in sections if s.amended)
    removed = sum(len(s.removed_text) for s in sections)
    print(
        f"\ncorpus     {len(sections)} sections, {chars:,} chars"
        f"\namended    {amended} sections changed this edition, {removed} passages dropped"
        f"\neval set   {len(faqs):,} Q&A pairs"
        f"\nwritten to {OUT_DIR}/"
    )


if __name__ == "__main__":
    main()
