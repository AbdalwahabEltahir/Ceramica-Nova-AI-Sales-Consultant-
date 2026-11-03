"""Hybrid retrieval: BM25 + dense cosine similarity, merged by reciprocal
rank fusion (RRF) — BM25 catches exact terms (SKU, 'R12'), dense catches
paraphrases and Arabic↔English matches. Falls back to BM25-only when
embeddings aren't configured or a call failed (see embeddings_client.py).
"""

from rag import lancedb_store
from rag.embeddings_client import embed_text, is_configured
from rag.index import get_index

RRF_K = 60  # standard RRF smoothing constant
DEFAULT_TOP_K = 8


def _metadata_passes(doc, metadata_filter):
    if not metadata_filter:
        return True
    meta = doc.get('metadata', {})
    for key, value in metadata_filter.items():
        if meta.get(key) != value:
            return False
    return True


def retrieve(query, top_k=DEFAULT_TOP_K, metadata_filter=None):
    """Returns a list of {'id', 'type', 'text', 'metadata', 'score', 'citation'}
    ranked by fused BM25 + dense relevance, restricted to documents that pass
    metadata_filter (the FR-3.2 pre-filter — hard constraints still come from
    models/rules.py, this is just document eligibility for retrieval)."""
    index = get_index()
    eligible = [i for i, doc in enumerate(index.documents) if _metadata_passes(doc, metadata_filter)]
    if not eligible:
        return []

    bm25_hits = index.bm25.search(query, top_k=max(top_k * 2, 10))
    bm25_rank = {doc_i: rank for rank, (doc_i, _score) in enumerate(bm25_hits) if doc_i in eligible}

    dense_rank = {}
    if is_configured():
        query_vec = embed_text(query)
        if query_vec is not None:
            eligible_ids = {index.documents[i]['id'] for i in eligible}
            id_to_index = {index.documents[i]['id']: i for i in eligible}
            hits = lancedb_store.search(query_vec, top_k=max(top_k * 2, 10), doc_ids=eligible_ids)
            dense_rank = {id_to_index[doc_id]: rank for rank, doc_id in enumerate(hits)}

    fused_scores = {}
    for doc_i in set(bm25_rank) | set(dense_rank):
        score = 0.0
        if doc_i in bm25_rank:
            score += 1.0 / (RRF_K + bm25_rank[doc_i] + 1)
        if doc_i in dense_rank:
            score += 1.0 / (RRF_K + dense_rank[doc_i] + 1)
        fused_scores[doc_i] = score

    ranked = sorted(fused_scores.items(), key=lambda x: -x[1])[:top_k]

    results = []
    for doc_i, score in ranked:
        doc = index.documents[doc_i]
        citation = doc['metadata'].get('sku') or doc['metadata'].get('rule_id') or doc['id']
        results.append({
            'id': doc['id'],
            'type': doc['type'],
            'text': doc['text'],
            'metadata': doc['metadata'],
            'score': round(score, 5),
            'citation': citation,
        })
    return results


def retrieval_relevance_score(query, sku):
    """0..1 style relevance score for one SKU against a query — used as the
    'retrieval relevance' soft-ranking signal in services/recommender.py."""
    results = retrieve(query, top_k=10)
    for rank, result in enumerate(results):
        if result['metadata'].get('sku') == sku:
            return max(0.0, 1.0 - rank / 10)
    return 0.0
