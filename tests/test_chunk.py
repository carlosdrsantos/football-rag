"""Chunker regressions and corpus integrity."""

import json
import re
from pathlib import Path

import pytest

from lotg.ingest.chunk import MAX_CHARS, Chunk, chunk_section, embed_text

PROCESSED = Path("data/processed")

SECTION = {
    "id": "law-12-12.4",
    "law_number": 12,
    "law_title": "Fouls and Misconduct",
    "clause": "12.4",
    "clause_title": "Disciplinary action",
    "url": "https://www.theifab.com/laws/latest/fouls-and-misconduct/",
    "markdown": (
        "## 12.4 Disciplinary action\n\n"
        "The referee has the power to take disciplinary action.\n\n"
        "### PLAYERS, SUBSTITUTES AND SUBSTITUTED PLAYERS\n\n"
        "### Sending-off offences\n\n"
        "A player is sent off for serious foul play.\n\n"
        "### TEAM OFFICIALS\n\n"
        "The senior coach receives the sanction.\n\n"
        "### Sending-off\n\n"
        "A team official is sent off for entering the field.\n"
    ),
}


def _by_subsection(chunks: list[Chunk], name: str) -> Chunk:
    return next(c for c in chunks if c.subsection == name)


def test_all_caps_heading_becomes_a_group():
    chunks = chunk_section(SECTION)
    assert _by_subsection(chunks, "Sending-off offences").group == (
        "PLAYERS, SUBSTITUTES AND SUBSTITUTED PLAYERS"
    )
    assert _by_subsection(chunks, "Sending-off").group == "TEAM OFFICIALS"


def test_breadcrumb_distinguishes_player_from_team_official():
    chunks = chunk_section(SECTION)
    player = _by_subsection(chunks, "Sending-off offences").breadcrumb
    official = _by_subsection(chunks, "Sending-off").breadcrumb
    assert player.endswith("PLAYERS, SUBSTITUTES AND SUBSTITUTED PLAYERS > Sending-off offences")
    assert official.endswith("TEAM OFFICIALS > Sending-off")


def test_group_heading_with_its_own_body_is_kept():
    chunk = next(c for c in chunk_section(SECTION) if c.subsection is None and c.group)
    assert chunk.group == "TEAM OFFICIALS"
    assert "senior coach" in chunk.body


def test_empty_body_produces_no_chunk():
    assert all(c.body.strip() for c in chunk_section(SECTION))


def test_preamble_is_kept():
    intro = next(c for c in chunk_section(SECTION) if c.id.endswith("#intro"))
    assert "power to take disciplinary action" in intro.body
    assert intro.group is None


def test_embed_text_leads_with_the_breadcrumb():
    chunk = _by_subsection(chunk_section(SECTION), "Sending-off")
    assert embed_text(chunk).startswith("Law 12 Fouls and Misconduct > 12.4")
    assert chunk.body in embed_text(chunk)


def test_long_bodies_split_between_paragraphs():
    paragraph = "Sentence one is here. " * 12
    section = SECTION | {"markdown": "## 1.1 Long\n\n" + "\n\n".join([paragraph] * 12)}
    chunks = chunk_section(section)
    assert len(chunks) > 1
    for chunk in chunks:
        assert len(chunk.body) <= MAX_CHARS
        assert chunk.body.strip().endswith(".")


def test_split_parts_get_distinct_ids():
    section = SECTION | {"markdown": "## 1.1 Long\n\n" + "\n\n".join(["word " * 100] * 12)}
    ids = [c.id for c in chunk_section(section)]
    assert len(ids) == len(set(ids))


def _load(name: str) -> list[dict]:
    path = PROCESSED / name
    if not path.exists():
        pytest.skip(f"{path} missing, run `make fetch parse chunk`")
    return [json.loads(line) for line in path.open(encoding="utf-8")]


def _normalise(text: str) -> str:
    return re.sub(r"\s+", "", re.sub(r"^#+ .*$", "", text, flags=re.MULTILINE))


def test_chunking_loses_no_text():
    bodies: dict[str, list[str]] = {}
    for chunk in _load("chunks.jsonl"):
        bodies.setdefault(chunk["section_id"], []).append(chunk["body"])

    for section in _load("sections.jsonl"):
        joined = _normalise("\n\n".join(bodies.get(section["id"], [])))
        assert _normalise(section["markdown"]) == joined, section["id"]


def test_real_chunk_ids_are_unique():
    ids = [c["id"] for c in _load("chunks.jsonl")]
    assert len(ids) == len(set(ids))


def test_no_real_chunk_exceeds_the_cap():
    oversized = [c["id"] for c in _load("chunks.jsonl") if len(c["body"]) > MAX_CHARS]
    assert not oversized, oversized


def test_every_breadcrumb_starts_with_its_law():
    for chunk in _load("chunks.jsonl"):
        assert chunk["breadcrumb"].startswith(f"Law {chunk['law_number']} ")
