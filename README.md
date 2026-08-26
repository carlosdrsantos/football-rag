# Laws of the Game RAG

Question answering over the IFAB *Laws of the Game*, with a citation to the
clause that actually decides the question.

The football is the fun part. The engineering problem is retrieval over a dense,
cross-referencing, versioned rulebook, where the failure that hurts is not a
missing answer but a confident one citing the wrong clause.

## Status

Ingestion, chunking, reranked hybrid retrieval, an eval harness and an HTTP
service. `/search` needs no API key. `/ask` generates a cited ruling and needs
one.

```
corpus      87 sections -> 139 chunks, 104,803 chars, all 17 Laws
chunks      median 642 chars, max 2,057, none over the 2,000 cap
amended     19 sections changed this edition, 22 repealed passages dropped
eval set    789 official IFAB Q&A rows, 595 distinct questions after collapsing
retrieval   recall@1 73.4%, @5 96.3%, @10 98.8%  (BM25 + bge-small, RRF, cross-encoder)
```

## The corpus is the Laws, the eval set is IFAB's Q&A

Every Law page ships official question-and-answer pairs, each one a realistic
refereeing scenario with the authoritative ruling. There are 789 of them, 595
once the cross-listings collapse. That is an eval set written by the people who
write the rules, phrased the way referees actually ask, against the 30-odd
hand-written questions a project like this usually gets.

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

That fixes the cross-listed questions. It does nothing for the ones IFAB filed
only under the Law describing the scenario while the ruling lives elsewhere, so
every one of the 595 was then read against its official answer. 23 needed a
change, and `data/labels/overrides.jsonl` records each with the reason:

```json
{"id": "65c174a2", "gold": [3, 8],
 "note": "dog is an outside agent, restart is a dropped ball in the penalty area (Law 8.2)"}
```

Everything not in that file keeps IFAB's own filing, so the labels are a diff
against the source rather than a replacement for it, and the 23 judgement calls
are the only thing a reader has to audit. `test_every_override_matches_a_real_question`
fails the build on a mistyped id, which would otherwise relabel nothing in
silence.

All three keys are scored in the same pass off the same retrieved lists, so
nothing but the key differs between the columns.

## Two retrievers, fused on rank

Dense retrieval was measured first, alone, so the hybrid had something to beat.

The dense model loses on rare terms, which is most of what separates the small
Laws. So the second retriever is BM25, and specifically BM25 rather than Postgres
full-text search: `ts_rank` and `ts_rank_cd` weight term frequency and coverage
but carry **no IDF at all**, and IDF is the entire reason for adding a lexical
leg. "Defective ball" has to outweigh "player" and only IDF does that. It is
about 60 lines over 139 chunks, stemmed with snowball, no stopword list because
a term in every chunk already earns an IDF of nearly zero.

Cosine similarity and a BM25 score have nothing to say to each other, so the two
are fused on **rank**, not score. Reciprocal rank fusion gives a chunk
1 / (60 + rank) from each list it appears in and sums.

| k | dense | BM25 | hybrid | reranked | random | lift |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 63.7% | 66.6% | 69.1% | **73.4%** | 14.0% | 5.3x |
| 3 | 84.4% | 86.6% | 89.7% | **90.3%** | 34.7% | 2.6x |
| 5 | 92.9% | 92.6% | 94.5% | **96.3%** | 48.7% | 2.0x |
| 10 | 97.5% | 98.0% | 98.8% | 98.8% | 68.4% | 1.4x |

This time the lift moved too, 4.6x to 5.3x at k=1. The answer key did not change,
so unlike the earlier rounds the retriever genuinely got better.

BM25 alone beating the embedding model at k=1 was not what I expected. A 384-dim
model on a rulebook full of exact terms of art is not obviously the stronger leg,
and on this corpus it is not.

**RRF k was not tuned.** Sweeping it over 10, 20, 60 and 120, against candidate
depths of 10 through 100, moves recall@1 between 68.4% and 69.2%. There is
nothing there to fit, so it keeps the value from the paper. Indexing the
breadcrumb alongside the body is worth having though: BM25 scores 66.6% on
breadcrumb plus body against 65.2% on body alone.

The older answer keys are still scored every run, so a change to the key can
never be mistaken for a change to the retriever. The reranked stack scores 72.3%
at k=1 under IFAB's raw cross-listing and 54.6% under one-Law-per-row.

## The per-Law table was asking the wrong question

I read the first per-Law breakdown as "Law 9 retrieval is broken at 5.3%". It
was measuring whether a Law *itself* appeared at k=1, which is not whether the
question got answered. Once gold sets hold more than one Law those come apart,
and they come apart a long way (hybrid, before reranking):

| Law | n | answered@1 | Law shown@1 |
|---:|---:|---:|---:|
| 5 | 76 | 80.3% | 34.2% |
| 6 | 7 | 85.7% | 28.6% |
| 9 | 19 | 47.4% | 10.5% |
| 14 | 73 | 74.0% | 41.1% |
| 16 | 20 | 35.0% | 10.0% |
| 2 | 2 | 0.0% | 0.0% |

Law 9 defines when the ball is in and out of play, 977 chars in two chunks. Its
questions ask what happens when the ball hits the referee, and IFAB's own answer
is "dropped ball", which is Law 8.2. Retrieval returns Law 8.2, which is right,
and Law 9 never surfaces. 47% of those questions are answered, not 10%.

Both columns are worth keeping, because they fail for different reasons and only
one of them is a retrieval problem. `answered` is the headline metric restricted
to one Law's traffic. `Law shown` is what citation precision needs, and a Law
that never surfaces cannot be cited even when the answer is correct.

## What hybrid cost, and the one it could not fix

It is a net win, not a uniform one. Law 12 goes 55.8% to 66.6% answered@1 and
Law 6 goes 71.4% to 85.7%, but Law 8 drops 72.2% to 59.3% and Law 3 drops 76.0%
to 72.0%. BM25 is much weaker on those two (51.9% and 54.0%) and RRF weights both
legs equally, so the weaker one drags. Weighting the fusion would recover it and
would also be a knob fitted to these 595 questions, which is why it has not been
touched.

Law 2 still answers neither of its two questions, and no retriever will fix it.
Both are phrased "the ball bursts". **The word "burst" does not appear once in
the corpus.** The Laws say "defective":

> If the ball becomes defective: play is stopped and restarted with a dropped ball

Ask the same question in the rulebook's own words and hybrid returns Law 2.2 at
rank 1. That gap is not a ranking problem and more retrieval tuning cannot close
it, which makes it the argument for query expansion or a reranker rather than for
another retriever. Two questions is also not enough to conclude anything, so it
is written down as an observation, not a result.

## A cross-encoder over the top ten

Both retrievers above score a question against a chunk without ever seeing the
two together. The chunk becomes 384 numbers, or a bag of stemmed terms, before
the question arrives. A cross-encoder reads the pair as one sequence, which is
why it can tell that "when can a coach be sent off" wants TEAM OFFICIALS and not
PLAYERS. It costs a forward pass per candidate, so it reranks rather than
searches: `BAAI/bge-reranker-base` over what hybrid already found.

**Depth 10, chosen from the curve rather than taste.** Hybrid recall@10 is 98.8%,
so the answer is nearly always inside what the cross-encoder gets to see. Past
depth 5 the curve is flat, and paying for 25 buys nothing:

| depth | @1 | @5 | ms/query |
|---:|---:|---:|---:|
| 0 | 69.1% | 94.5% | 0 |
| 5 | 73.6% | 94.5% | 146 |
| **10** | **73.4%** | **96.3%** | **291** |
| 15 | 73.9% | 96.1% | 437 |
| 25 | 73.3% | 96.1% | 728 |

The spread from 5 to 25 is four questions out of 595, which is noise. Depth 10 is
where recall@5 tops out, and recall@5 is what a generator would actually be fed.

**A cheaper cross-encoder is worse than none at all.** `ms-marco-MiniLM-L-6-v2`
is 23M parameters against 278M and runs in 138 ms instead of 728, and it scores
68.4% at k=1: *below* the 69.1% of the hybrid it was reranking. It reorders
confidently and wrongly. That is the useful half of the comparison, because the
small model is the one you would reach for on latency grounds.

## What reranking fixed, and what it broke

It went after exactly the failure it was picked for. The small procedural Laws
were losing because the right clause sat at rank 3 or 4, not because it was
missing, which is the definition of a reranking problem:

| Law | n | answered@1 | Law shown@1 |
|---:|---:|---:|---:|
| 16 | 20 | 35.0% -> **65.0%** | 10.0% -> **45.0%** |
| 9 | 19 | 47.4% -> 52.6% | 10.5% -> **36.8%** |
| 6 | 7 | 85.7% -> 85.7% | 28.6% -> **71.4%** |
| 17 | 8 | 62.5% -> 62.5% | 25.0% -> **50.0%** |
| 12 | 308 | 66.6% -> 74.0% | 51.9% -> 57.8% |

The query the chunking work was built around lands too. `when can a coach be sent
off` needs Law 12.4 > TEAM OFFICIALS > Sending-off, not the identically titled
one under PLAYERS:

```
hybrid     1. Law 3  > 3.6 Players and substitutes sent off
           2. Law 12 > 12.4 Disciplinary action > TEAM OFFICIALS
reranked   1. Law 12 > 12.4 Disciplinary action > TEAM OFFICIALS
           2. Law 5  > 5.3 Powers and duties > Disciplinary action
```

The `Law shown` column is the one to watch. Law 6 answers the same share of its
questions either way, but the Law itself now surfaces at rank 1 for 71% of them
instead of 29%, and a Law that never surfaces cannot be cited. Citation precision
is the stated point of the project and it is the column that moved most.

It is not free. Law 11 drops 86.7% to 80.0% answered@1 across 75 questions, Law
10 drops 79.3% to 72.4%, and Law 1 drops 87.5% to 62.5% on 8 questions. Offside
is the clearest loss: the dense model was already good at it and the cross-encoder
talks it out of correct answers. Net it is worth 4.3 points, and the losses are
recorded because a headline number that hides them is worth less than the points
it claims.

Law 2 still answers neither of its questions, exactly as predicted. Reranking
cannot invent the word "burst".

The random column is computed exactly, not sampled: for a gold set covering n of
the N chunks, k blind draws miss all of it with probability C(N-n, k) / C(N, k).
There is a second baseline worth knowing too. Law 12 is in 308 of the 595 gold
sets, so a system that always answered from Law 12 scores 51.8%.

That column earned its place on the two rounds before this one. Fixing the answer
key moved dense recall@1 from 47.3% to 63.7% and moved the lift by nothing: a
gold set of three Laws is a wider target for a blind draw too, and the random
column absorbed the whole gain. The retriever had not improved, the question had
got easier, and the lift is what said so.

Reading all 595 answers by hand was worth 1.2 of those points. IFAB's own
cross-listing was doing almost all the work and the 23 overrides mostly confirmed
the filing rather than corrected it. Not the result I expected, and the useful
kind of negative: the key is audited rather than assumed, which is what makes
these hybrid numbers worth anything.

## The service

Two endpoints, split by what they need.

`POST /search` is the retrieval stack over HTTP and needs no API key. `POST /ask`
puts a model on top and returns a ruling whose every claim carries a clause
citation. `GET /health` reports whether the index has anything in it and which
models are loaded, including a null generator when no key is set.

Both models and the BM25 index load once at startup. Steady-state `/search` is
about 340 ms end to end, of which roughly 290 is the cross-encoder. The first
request after boot is 8.3 s, because MPS compiles its graph on the first real
forward pass and warming the model object is not enough to trigger it. The
Dockerfile bakes both models into the image rather than pulling 420 MB during a
deploy.

**The model cites by position, not by clause number.** It sees `[1]` to `[5]` and
returns the numbers it used, which are mapped back to real chunks in
`cited_hits`. Asking it to reproduce `law-12-12.4#sending-off` invites a citation
that looks right and points nowhere, and a wrong integer is something code can
catch. Positions outside the range are dropped rather than rendered, and the
tests cover 0, -1 and 9 against a list of 3.

It also gets an explicit way out. `sufficient: false` says the clauses do not
decide the question, which is the honest answer often enough that not offering it
would guarantee invention.

Without a key `/ask` returns 503 with a reason rather than a 500:

```
$ curl -s -X POST localhost:8000/ask -d '{"question":"when can a coach be sent off"}'
{"detail":"ANTHROPIC_API_KEY is not set, so /ask cannot answer"}
```

## Measuring answers without wrecking the retrieval numbers

`evaluate_answers.py` is a separate file writing to a separate output, because it
breaks the two properties `make eval` depends on: it costs money and it is not
deterministic. Folding them together would quietly cost the retrieval numbers the
thing that makes them worth trusting.

Most of it still needs no judge. Whether the service abstained, whether it cited a
clause it was never shown, and whether the clause it cited belongs to a gold Law
are all set membership against labels that already exist. Only agreement with
IFAB's ruling needs a model, and that model is asked to compare two answers, never
to grade retrieval.

Two distinctions the metrics keep apart:

- **Abstaining with the answer in hand.** Refusing when nothing useful was
  retrieved is correct. Refusing when the gold Law was sitting in the prompt is
  a failure, and only the second is worth fixing.
- **Cited a gold Law** against **cited only gold Laws.** The first says the answer
  is grounded. The second says nothing extra was dragged in, and it is the
  stricter one.

The sample is every 6th question, roughly 100 of the 595, spread across all 17
Laws by the file order. Deterministic rather than random, so two runs measure the
same questions and can be compared.

Retrieval runs first and sequentially, because a psycopg connection is not safe
to share between threads and 100 local searches cost about 35 seconds anyway. The
two API calls per question depend on nothing else, so those go through a thread
pool, eight at a time. The first version ran all 200 calls one after another and
printed nothing until the end, which made a slow run and a hung one look
identical.

The judge defaults to `claude-sonnet-5` while the generator defaults to
`claude-opus-5`. Deciding whether two answers reach the same ruling, with both in
front of you, is the easy half and does not need the expensive model. It also
should not be the same model: one grading its own output rates it generously.

Which generator is worth paying for is a question to measure, not to assume, so
`--model` exists and the answer belongs in this table once it has been run.

**This has not been run yet.** It needs a key, and no numbers are reported here
until it has been.

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
  retrieval/bm25.py      BM25 with IDF, snowball stemming, no stopword list
  retrieval/reranker.py  bge-reranker-base cross-encoder
  retrieval/store.py     pgvector schema, insert, cosine search
  retrieval/retrieve.py  dense, BM25, RRF fusion and reranking behind one interface
  retrieval/build.py     chunks.jsonl -> pgvector
  retrieval/search.py    query any of the four from the shell
  service/app.py      FastAPI: /health, /search, /ask
  service/answer.py   the prompt, and the mapping from [n] back to a clause
evaluate_answers.py   citation and agreement eval -> evals/answers.json
  gold.py             FAQ rows -> one query per question, gold set of Laws
  evaluate.py         recall@k over those queries -> evals/baseline.json
data/labels/          the 23 relabelled questions, with a reason each
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
make eval        # all four retrievers -> evals/baseline.json, about 3.5 minutes
make serve       # http://localhost:8000/docs
make eval-answers ARGS="--limit 6 --model claude-haiku-4-5"   # cheap smoke test
make eval-answers  # the real thing: ~100 questions, needs ANTHROPIC_API_KEY
make search Q="when can a coach be sent off"
make search R=hybrid Q="..."    # or R=dense, R=lexical, to see what reranking changed
```

Embeddings, BM25 and the cross-encoder all run locally, so retrieval, `/search`
and `make eval` need no API key and reproduce on a fresh clone. The two models
download once, about 420 MB together. Only `/ask` and `make eval-answers` call
out to Anthropic.

## Roadmap

- [x] Ingestion, corpus and eval set kept apart
- [x] Chunking on `<h3>` boundaries with breadcrumbs, no text lost
- [x] Dense retrieval on pgvector, measured against exact random and
      majority-Law baselines
- [x] Eval harness. Recall@k over the FAQs, no LLM judge involved
- [x] Multi-Law gold sets from IFAB's own cross-listing, with the old key still
      reported next to the new one
- [x] All 595 answers read against their filing, 23 relabelled with a reason
      each, older keys still scored in the same pass
- [x] Hybrid BM25 and dense fused with RRF, every retriever scored in one run
- [x] Cross-encoder reranking of the top 10, with the depth and the model choice
      measured rather than assumed
- [x] FastAPI service and a Dockerfile, models baked into the image
- [x] Answer generation with citations, an abstention path, and an eval that
      keeps the LLM judge away from the retrieval numbers
- [ ] Run the generation eval and act on what it says
- [ ] Eval in CI, blocking merges on retrieval regression
- [ ] Deploy to Azure
- [ ] Query expansion, if the vocabulary gap ever shows up in more than 2 questions

## Known gaps

- One edition only. Handling version collisions needs a prior edition to compare
  against, and the IFAB does not serve old ones under `/laws/latest/`.
- Seven `Introduction` sections have no clause number and can only be cited to
  the Law. Correct, but it caps citation precision for them.
- The eval set is lopsided. Law 12 is in 308 of the 595 gold sets, Law 2 in 2.
  Any headline number needs the per-Law breakdown next to it or Law 12 swamps it.
- The generation half is unmeasured. Every claim about citation quality and
  abstention is a design intention until `make eval-answers` has been run once.
- Nothing rate-limits or authenticates `/ask`, so anyone who can reach the
  service can spend the key behind it. That is fine on localhost and is the
  first thing to fix before it is reachable from anywhere else.
- Reranking costs about 291 ms a query on Apple Silicon GPU, against roughly 20
  for hybrid alone. That is the whole latency budget of the service and it is
  spent before a single token is generated. It also makes `make eval` take three
  and a half minutes instead of seconds.
- The cross-encoder runs on whatever device torch picks, and MPS and CPU do not
  produce bit-identical floats. Near-ties can order differently across machines,
  so a recall number could move by a question or two on a different box.
- BM25 runs in process and is rebuilt from `chunks.jsonl` on startup. At 139
  chunks that is free and the arithmetic is mine to test, but it is the piece
  that has to move into the database or a search service before the corpus is
  large enough to matter.
- Fusion weights both retrievers equally. Law 8 and Law 3 would be better served
  by leaning on the dense leg, but tuning that against these same 595 questions
  is how you end up reporting a number that only holds on your own eval set.
- The 23 relabels are one person's reading of one official answer each. The
  reasons are committed so they can be argued with, but nobody has second-read
  them, and two questions is not enough to say anything about Law 2.
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
