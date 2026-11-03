"""Minimal BM25 (Okapi) over tokenised documents — no external dependency.
Catches exact terms (SKU codes, 'R12', '60x120') that dense embeddings can blur."""

import math
import re
from collections import Counter

TOKEN_RE = re.compile(r"[\w؀-ۿ]+", re.UNICODE)

# Without stopword removal, every document shares "is"/"a"/"the"/"of" with
# every query, so BM25 scores every candidate near-equally regardless of
# actual relevance (found via a real query: "frost rating of CN003?" scored
# an unrelated business-rule doc almost the same as the CN003 doc itself).
STOPWORDS = {
    'a', 'an', 'the', 'is', 'are', 'was', 'were', 'be', 'been', 'of', 'in',
    'on', 'for', 'to', 'and', 'or', 'with', 'this', 'that', 'it', 'its',
    's', 't', 'do', 'does', 'what', 'which', 'you', 'your', 'i', 'me', 'my',
    'can', 'could', 'about', 'tell', 'please',
    'في', 'من', 'الى', 'إلى', 'على', 'هل', 'ما', 'هذا', 'هذه', 'أن', 'و',
}


def tokenise(text):
    return [t.lower() for t in TOKEN_RE.findall(text or '') if t.lower() not in STOPWORDS]


class BM25Index:
    def __init__(self, documents, k1=1.5, b=0.75):
        self.k1 = k1
        self.b = b
        self.documents = documents
        self.tokenised = [tokenise(doc['text']) for doc in documents]
        self.doc_len = [len(toks) for toks in self.tokenised]
        self.avg_len = sum(self.doc_len) / len(self.doc_len) if self.doc_len else 0
        self.df = Counter()
        for toks in self.tokenised:
            for term in set(toks):
                self.df[term] += 1
        self.n_docs = len(documents)

    def _idf(self, term):
        df = self.df.get(term, 0)
        return math.log(1 + (self.n_docs - df + 0.5) / (df + 0.5))

    def score(self, query_tokens, doc_index):
        toks = self.tokenised[doc_index]
        freqs = Counter(toks)
        dl = self.doc_len[doc_index]
        score = 0.0
        for term in query_tokens:
            if term not in freqs:
                continue
            idf = self._idf(term)
            f = freqs[term]
            denom = f + self.k1 * (1 - self.b + self.b * dl / (self.avg_len or 1))
            score += idf * (f * (self.k1 + 1)) / (denom or 1)
        return score

    def search(self, query, top_k=8):
        query_tokens = tokenise(query)
        scored = [(i, self.score(query_tokens, i)) for i in range(self.n_docs)]
        scored = [s for s in scored if s[1] > 0]
        scored.sort(key=lambda x: -x[1])
        return scored[:top_k]
