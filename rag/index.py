"""Builds the hybrid (BM25 + dense) index over the sanitised corpus. Dense
embeddings are cached to disk since the free HF Inference API is slow/rate
limited — we only want to pay that cost once per corpus version, not per
request (NFR-8 caching: cache embeddings, never cache inventory/quotes).
"""

import json
import os

from backend.app.services.guardrails.content_sanitizer import sanitise_corpus
from rag import lancedb_store
from rag.bm25 import BM25Index
from rag.documents import build_documents
from rag.embeddings_client import embed_texts, is_configured, model_id

CACHE_PATH = os.path.join(os.path.dirname(__file__), '.embeddings_cache.json')


class RagIndex:
    def __init__(self, documents, bm25, dense_vectors, quarantined):
        self.documents = documents
        self.bm25 = bm25
        self.dense_vectors = dense_vectors  # list aligned with documents, entries may be None
        self.quarantined = quarantined


def _load_cache():
    if os.path.exists(CACHE_PATH):
        try:
            with open(CACHE_PATH, 'r', encoding='utf-8') as fh:
                return json.load(fh)
        except (OSError, json.JSONDecodeError):
            return {}
    return {}


def _save_cache(cache):
    try:
        with open(CACHE_PATH, 'w', encoding='utf-8') as fh:
            json.dump(cache, fh)
    except OSError:
        pass


def build_index(include_fixture=None):
    """include_fixture: optional list of extra raw documents (e.g. the C3
    malicious-record fixture) to run through the same quarantine step, for
    demos/tests. Not used in the live index by default."""
    documents = build_documents()
    if include_fixture:
        documents = documents + include_fixture

    clean, quarantined = sanitise_corpus(documents)

    bm25 = BM25Index(clean)

    dense_vectors = [None] * len(clean)
    if is_configured():
        model = model_id()
        cache = _load_cache()
        to_embed = []
        to_embed_indices = []
        for i, doc in enumerate(clean):
            cached = cache.get(doc['id'])
            if cached and cached.get('text') == doc['text'] and cached.get('model') == model:
                dense_vectors[i] = cached['vector']
            else:
                to_embed.append(doc['text'])
                to_embed_indices.append(i)

        if to_embed:
            vectors = embed_texts(to_embed, task='document')
            for idx, vec in zip(to_embed_indices, vectors):
                dense_vectors[idx] = vec
                if vec is not None:
                    cache[clean[idx]['id']] = {'text': clean[idx]['text'], 'model': model, 'vector': vec}
            _save_cache(cache)

        lancedb_store.sync(clean, dense_vectors, model)

    return RagIndex(clean, bm25, dense_vectors, quarantined)


_index_cache = None


def get_index(force_rebuild=False):
    global _index_cache
    if _index_cache is None or force_rebuild:
        _index_cache = build_index()
    return _index_cache
