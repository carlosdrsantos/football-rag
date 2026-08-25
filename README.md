# Laws of the Game RAG

Question answering over the IFAB *Laws of the Game*, with a citation to the
clause that actually decides the question.

The football is the fun part. The engineering problem is retrieval over a dense,
cross-referencing, versioned rulebook, where the failure that hurts is not a
missing answer but a confident one citing the wrong clause.

## Status

Ingestion works. Retrieval, serving and evaluation are next.

```
corpus      87 sections, 104,803 chars, all 17 Laws
amended     19 sections changed this edition, 22 repealed passages dropped
eval set    789 official IFAB Q&A pairs
```

## The corpus is the Laws, the eval set is IFAB's Q&A

Every Law page ships official question-and-answer pairs, each one a realistic
refereeing scenario with the authoritative ruling. There are 789 of them. That
is an eval set written by the people who write the rules, phrased the way
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

## Bugs found so far

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
tests/test_parse.py   parser regressions and corpus integrity
docker-compose.yml    Postgres 17 with pgvector on :5433
```

Fetch and parse are separate because chunking will be rewritten many times as
retrieval gets measured, and none of those passes should re-download the site.

## Running it

```bash
make install     # venv and dependencies
make fetch       # 17 pages -> data/raw/, ~5 MB, cached, 1 req/sec
make parse       # -> data/processed/sections.jsonl + faqs.jsonl
make test        # parser regressions and corpus integrity
make db          # Postgres with pgvector
```

## Roadmap

- [x] Ingestion, corpus and eval set kept apart
- [ ] Chunking. 11 of 87 sections run past 2,000 chars and Law 12.4 hits 13,901,
      so they need splitting on `<h3>` boundaries without orphaning a subsection
      from its clause
- [ ] Baseline dense retrieval on pgvector, kept simple, measured before it is
      improved
- [ ] Eval harness. Recall@k over the 789 FAQs, scored on whether the right Law
      came back, which is an objective number rather than an LLM-judged one
- [ ] Hybrid BM25 and dense, then reranking, re-running the eval on each change
- [ ] FastAPI service, Docker, Azure
- [ ] Eval in CI, blocking merges on retrieval regression

## Known gaps

- One edition only. Handling version collisions needs a prior edition to compare
  against, and the IFAB does not serve old ones under `/laws/latest/`.
- Seven `Introduction` sections have no clause number and can only be cited to
  the Law. Correct, but it caps citation precision for them.
- The 789 FAQs are lopsided, 296 for Law 12 against 2 for Law 2. Any headline
  eval number needs a per-Law breakdown next to it or Law 12 swamps it.

## Source

Laws of the Game © The IFAB, from [theifab.com](https://www.theifab.com/laws/latest/).
Fetched for personal, non-commercial use, rate-limited to 1 request/second and
cached locally.
