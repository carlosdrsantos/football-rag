# Laws of the Game RAG

Question answering over the IFAB *Laws of the Game*, with a citation to the
clause that actually decides the question.

The football is the fun part. The engineering problem is retrieval over a dense,
cross-referencing, versioned rulebook, where the failure that hurts is not a
missing answer but a confident one citing the wrong clause.

## Status

Ingestion, chunking, dense retrieval and the eval harness work. Serving is next.

```
corpus      87 sections -> 139 chunks, 104,803 chars, all 17 Laws
chunks      median 642 chars, max 2,057, none over the 2,000 cap
amended     19 sections changed this edition, 22 repealed passages dropped
eval set    789 official IFAB Q&A rows, 595 distinct questions after collapsing
retrieval   recall@1 62.5%, @5 92.1%, @10 97.1%  (bge-small-en-v1.5, 384d, pgvector)
```

## The corpus is the Laws, the eval set is IFAB's Q&A

Every Law page ships official question-and-answer pairs, each one a realistic
refereeing scenario with the authoritative ruling. There are 789 of them, 595
once the cross-listings collapse. That is an eval set written by the people who
write the rules, phrased the way
referees actually ask, against the 30-odd hand-written questions a project like
this usually gets.

It is worth nothing if the answers are in the index. A question would retrieve
its own answer, every metric would sit at the ceiling, and none of them would
say anything about whether the Laws are retrievable. So `sections.jsonl` gets
embedded, `faqs.jsonl` is held out, and `tests/test_parse.py` fails the build if
question text turns up in the corpus.

That leaves the question worth measuring. Can retrieval over the Laws support a
correct ruling on a scenario it has never seen?

## Repealed law is not law

The IFAB marks each edition's amendments inline.

```html
<explanation data-explanation="rationale">
  <addedexplanation>text added this edition</addedexplanation>
  <deletedexplanation>text removed this edition</deletedexplanation>
</explanation>
```

`get_text()` flattens all of it into one string, splicing wording that was
repealed back into the current Laws. The answers that come out are fluent, well
cited, and describe a rule that no longer exists, which is worse than no answer
at all. The parser drops deleted wording from the indexed text and keeps it in
`removed_text`, where it is useful for building version-sensitive eval cases.

## Every chunk says where it came from

Sections run from 21 to 13,901 chars, so the long ones split at `<h3>`
boundaries and, failing that, between paragraphs. Four sections needed the
paragraph fallback, the ones written as continuous prose with no subheadings.

Size alone would have been the wrong rule twice.

Law 12.4 uses all-caps headings as group labels rather than rules. It files
"Sending-off offences" under PLAYERS and "Sending-off" under TEAM OFFICIALS, so
a chunk titled "Sending-off" cannot say whether it sends off a player or a
coach, and "when is a coach sent off" has to reach the second one. An all-caps
`<h3>` becomes a group and carries down to its siblings.

The clash is not confined to Law 12. "Procedure" appears as a subsection three
times across the corpus and "Advantage" twice. So every chunk leads with a
breadcrumb, and the breadcrumb is part of what gets embedded:

```
Law 12 Fouls and Misconduct > 12.4 Disciplinary action > TEAM OFFICIALS > Sending-off
```

`test_chunking_loses_no_text` reassembles every chunk back into its section and
compares against the parsed text, so a splitter that quietly drops a list item
fails the build.

## A question can be right about more than one Law

The first answer key was one Law per question, the Law whose page the question
appeared on. It was wrong, and IFAB's own pages say so. The same scenario is
published on every Law page it touches, so the 789 rows are 595 distinct
questions and 171 of those carry two or three Laws:

> *A defender recklessly fouls an attacker and the point of contact with the
> attacker's leg is on the penalty area line. What is the decision?*

That one is filed under Law 1, Law 12 and Law 14. Under the old key it was three
separate questions, each with two of its three correct answers marked wrong. So
a question is now one query, and its gold set is every Law IFAB filed it under.

`gold.py` does the collapse and nothing else. The old score is still computed in
the same pass, off the same retrieved lists, so nothing but the key differs
between the two columns.

## The baseline

Dense retrieval only: one embedding model, cosine distance over pgvector, no
hybrid search and no reranking. Measured first so later changes have something
to move.

| k | recall | random | lift | old key | old random | old lift |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 62.5% | 13.5% | 4.6x | 47.3% | 10.2% | 4.6x |
| 3 | 83.2% | 33.7% | 2.5x | 67.6% | 26.6% | 2.5x |
| 5 | 92.1% | 47.5% | 1.9x | 78.7% | 39.0% | 2.0x |
| 10 | 97.1% | 67.3% | 1.4x | 89.7% | 58.7% | 1.5x |

The random column is computed exactly, not sampled: for a gold set covering n of
the N chunks, k blind draws miss all of it with probability C(N-n, k) / C(N, k).
There is a second baseline worth knowing too. Law 12 is in 296 of the 595 gold
sets, so a system that always answered from Law 12 scores 49.7%.

**Fixing the key moved recall 15 points and moved the lift by nothing.** Three
gold Laws instead of one is a wider target for a blind draw as well, and the
random column absorbs the whole gain. The retriever did not get better, the
question got easier, and the only column that noticed is the one that is
supposed to. Any single number here is worthless without it.

## Law 9 is still broken and the reason is still the key

Law 9 scores 5.3% at k=1 across 19 questions, unchanged by the collapse. It
defines when the ball is in and out of play, 977 chars in two chunks, and its
questions look like this:

> *A player plays the ball, which touches the referee, stays on the field, and
> possession changes. What is the decision?*

IFAB files that under Law 9 only. Its own answer is "play restarts with a dropped
ball", which is Law 8.2. Retrieval returns Law 8.2 Dropped ball and the metric
scores it wrong. Cross-listing does not save these because IFAB never filed them
anywhere else.

There is direct evidence for how common that is. 72 of the answers name a Law
outright, and in 30 of them the named Law is not one the question is filed under.
The small definitional Laws are where it bites: Law 2 scores 0% over 2 questions,
Law 16 11%, Law 6 14%, Law 5 26%.

The remaining fix is to read the gold Law out of the answer text rather than the
filing. Doing that automatically means either lexical matching, which would
flatter the BM25 half of a hybrid retriever measured against it, or an LLM
labelling pass written to disk once and committed. The second keeps the eval
itself free of any model, which is the property worth protecting.

## Bugs found so far

**`register_vector` failed before the extension existed.** The store created
the `vector` extension inside its schema DDL, but `connect()` called
`register_vector` first, which looks the type up by OID and found nothing.
`connect()` now creates the extension itself, so the type is guaranteed before
anything asks for it.

**Clause headings were being deleted silently.** The parser dropped `<button>`
elements as chrome. On theifab.com every clause heading sits inside the collapse
toggle, so all 87 sections lost their number and title. The corpus parsed
cleanly, the text read fine, and not one section could be cited. Spotted in the
ids, not the prose: `law-12-s4` should have been `law-12-12.4`. Buttons now
unwrap instead of decompose, and
`test_clause_number_survives_the_collapse_button` keeps it that way.

## Why the HTML and not the PDF

PDF parsing is the usual first instinct and it is the wrong call here. A precise
citation depends on the clause hierarchy, Law 12 > 12.4 > *Cautionable
offences*, and in the PDF that hierarchy exists only as font sizes and page
coordinates you have to reverse-engineer. The HTML is server-rendered with real
`<article>` and `<h2>`/`<h3>` tags, so the structure is read rather than
guessed, and it carries the amendment markup the PDF never exposes.

## Layout

```
src/lotg/
  sources.py          the 17 Law pages and their URLs
  ingest/fetch.py     download to data/raw/ (cached)
  ingest/parse.py     HTML -> sections.jsonl (corpus) + faqs.jsonl (eval set)
  ingest/chunk.py     sections.jsonl -> chunks.jsonl, the unit that gets embedded
  retrieval/embedder.py  bge-small-en-v1.5; queries get the instruction prefix
  retrieval/store.py     pgvector schema, insert, cosine search
  retrieval/build.py     chunks.jsonl -> pgvector
  retrieval/search.py    query it from the shell
  gold.py             FAQ rows -> one query per question, gold set of Laws
  evaluate.py         recall@k over those queries -> evals/baseline.json
tests/                parser regressions, chunker, eval maths, corpus integrity
docker-compose.yml    Postgres 17 with pgvector on :5433
```

Fetch and parse are separate because chunking will be rewritten many times as
retrieval gets measured, and none of those passes should re-download the site.

## Running it

```bash
make install     # venv and dependencies
make fetch       # 17 pages -> data/raw/, ~5 MB, cached, 1 req/sec
make parse       # -> data/processed/sections.jsonl + faqs.jsonl
make chunk       # -> data/processed/chunks.jsonl
make test        # parser regressions and corpus integrity
make db          # Postgres with pgvector
make index       # embed the chunks into pgvector
make eval        # recall@k over the 595 questions -> evals/baseline.json
make search Q="when can a coach be sent off"
```

Embeddings run locally through sentence-transformers, so there is no API key to
set and the eval reproduces on a fresh clone.

## Roadmap

- [x] Ingestion, corpus and eval set kept apart
- [x] Chunking on `<h3>` boundaries with breadcrumbs, no text lost
- [x] Baseline dense retrieval on pgvector, measured against exact random and
      majority-Law baselines
- [x] Eval harness. Recall@k over the FAQs, no LLM judge involved
- [x] Multi-Law gold sets from IFAB's own cross-listing, with the old key still
      reported next to the new one
- [ ] Gold Laws read from the answer text, for the questions IFAB filed only
      under the Law describing the scenario
- [ ] Hybrid BM25 and dense, then reranking, re-running the eval on each change
- [ ] FastAPI service, Docker, Azure
- [ ] Eval in CI, blocking merges on retrieval regression

## Known gaps

- One edition only. Handling version collisions needs a prior edition to compare
  against, and the IFAB does not serve old ones under `/laws/latest/`.
- Seven `Introduction` sections have no clause number and can only be cited to
  the Law. Correct, but it caps citation precision for them.
- The eval set is lopsided. Law 12 is in 296 of the 595 gold sets, Law 2 in 2.
  Any headline number needs the per-Law breakdown next to it or Law 12 swamps it.
- Questions are collapsed on exact text. Two pages wording the same scenario
  differently stay separate, so 595 is an upper bound on the distinct count.
- Chunks do not overlap and the 2,000 char cap is a guess. Both are knobs to
  test now that there is a retrieval number to move.
- Law-level relevance is coarse, and multi-Law gold sets make it coarser: a hit
  anywhere in any of three Laws counts, and Law 12 alone spans 22 chunks. Clause
  level labels would be stricter and need annotation the FAQs do not provide.
- No answer generation yet, so nothing here measures faithfulness or citation
  accuracy. This is retrieval only.

## Source

Laws of the Game © The IFAB, from [theifab.com](https://www.theifab.com/laws/latest/).
Fetched for personal, non-commercial use, rate-limited to 1 request/second and
cached locally.
