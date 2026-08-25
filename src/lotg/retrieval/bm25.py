"""BM25 over the chunk text.

The dense model answers 30% of the Law 16 questions and neither of the Law 2
pair. Both are three-chunk Laws whose questions turn on a term that appears
almost nowhere else: "goal kick", "defective ball". IDF is the part of BM25 that
finds those, and it is the reason this is not Postgres full-text search, whose
ts_rank family weights term frequency and coverage but carries no IDF at all.

Terms are stemmed so "kicks" reaches "kick". Stopwords are not removed because
they do not need to be: a term in every chunk gets an IDF of nearly zero from
the smoothed formula below.
"""

import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from functools import cache
from math import log

import snowballstemmer

K1 = 1.5
B = 0.75
WORD = re.compile(r"[a-z0-9]+")


@cache
def _stemmer() -> snowballstemmer.stemmer:
    return snowballstemmer.stemmer("english")


def tokenize(text: str) -> list[str]:
    return _stemmer().stemWords(WORD.findall(text.lower()))


@dataclass(frozen=True)
class BM25:
    frequencies: list[Counter]
    lengths: list[int]
    average_length: float
    idf: dict[str, float]
    postings: dict[str, list[int]]

    def search(self, query: str, limit: int) -> list[tuple[int, float]]:
        """Document indices and scores, best first. Ties break on index."""
        scores: dict[int, float] = defaultdict(float)
        for term in tokenize(query):
            if term not in self.postings:
                continue
            for document in self.postings[term]:
                frequency = self.frequencies[document][term]
                length = self.lengths[document] / self.average_length
                saturation = frequency + K1 * (1 - B + B * length)
                scores[document] += self.idf[term] * frequency * (K1 + 1) / saturation

        ranked = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
        return ranked[:limit]


def build(documents: list[str]) -> BM25:
    frequencies = [Counter(tokenize(document)) for document in documents]
    lengths = [sum(counts.values()) for counts in frequencies]

    postings: dict[str, list[int]] = defaultdict(list)
    for index, counts in enumerate(frequencies):
        for term in counts:
            postings[term].append(index)

    total = len(documents)
    # Lucene's smoothed IDF, which stays positive for a term in every document
    # where the textbook form goes negative.
    idf = {
        term: log(1 + (total - len(documents_with) + 0.5) / (len(documents_with) + 0.5))
        for term, documents_with in postings.items()
    }
    return BM25(frequencies, lengths, sum(lengths) / total, idf, dict(postings))
