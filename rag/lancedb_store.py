"""LanceDB vector store for the RAG corpus (FR-3.2's "local vector index").
One row per document: every catalogue column (so the table can be browsed and
filtered directly, e.g. in LanceDB Studio) plus the embedded text, the
embedding model id and the vector. rag/retriever.py fuses its vector search
with BM25; if no vectors exist (no embedding key, or the call failed) the
retriever runs BM25-only.

Data lives in rag/.lancedb/, table `documents`. Rebuild with `python -m rag.ingest`.
"""

import logging
import os

logger = logging.getLogger('lancedb_store')

DB_PATH = os.path.join(os.path.dirname(__file__), '.lancedb')
TABLE_NAME = 'documents'

_PRODUCT_COLUMNS = {
    # column -> (metadata key, empty value for non-product rows)
    'sku': ('sku', ''),
    'name': ('name', ''),
    'tile_type': ('tile_type', ''),
    'size': ('size', ''),
    'application': ('application', ''),
    'finish': ('finish', ''),
    'slip_rating': ('slip_rating', ''),
    'price_per_m2': ('price_per_m2', 0.0),
    'stock_m2': ('stock_m2', 0.0),
    'tiles_per_box': ('tiles_per_box', 0),
    'style': ('style', ''),
    'floor_ok': ('floor_ok', False),
    'wall_ok': ('wall_ok', False),
    'outdoor_ok': ('outdoor_ok', False),
    'commercial': ('commercial', False),
    'rule_id': ('rule_id', ''),
}

_db = None


def _connect():
    global _db
    if _db is None:
        import lancedb
        _db = lancedb.connect(DB_PATH)
    return _db


def _row(doc, vec, model):
    meta = doc.get('metadata', {})
    row = {'doc_id': doc['id'], 'type': doc['type'], 'text': doc['text']}
    for column, (key, empty) in _PRODUCT_COLUMNS.items():
        value = meta.get(key)
        row[column] = type(empty)(value) if value is not None else empty
    row['embedding_model'] = model
    row['vector'] = [float(x) for x in vec]
    return row


def sync(documents, vectors, model):
    """Rebuilds the table from (document, vector) pairs aligned by index.
    Documents whose embedding failed are left out (BM25 still covers them).
    If *no* vectors came back, the existing table is kept rather than wiped.
    Returns the number of rows written."""
    rows = [_row(doc, vec, model) for doc, vec in zip(documents, vectors) if vec]
    if not rows:
        return 0
    _connect().create_table(TABLE_NAME, data=rows, mode='overwrite')
    return len(rows)


def _open():
    db = _connect()
    if TABLE_NAME not in db.table_names():
        return None
    return db.open_table(TABLE_NAME)


def search(query_vector, top_k=20, doc_ids=None):
    """Returns doc_id strings ranked by vector similarity (closest first),
    restricted to doc_ids when given (the retriever's eligibility set)."""
    if not query_vector:
        return []
    table = _open()
    if table is None:
        return []
    n_rows = table.count_rows()
    if n_rows == 0:
        return []

    try:
        hits = (table.search(list(query_vector))
                .metric('cosine')
                .limit(min(top_k * 4, n_rows))
                .select(['doc_id'])
                .to_list())
    except Exception as exc:
        # e.g. the table was built with a different embedding model/dimension
        # and hasn't been re-ingested yet — dense is optional, BM25 carries on.
        logger.info('LanceDB search failed, BM25-only for this query: %s', exc)
        return []

    out = []
    for row in hits:
        if doc_ids is not None and row['doc_id'] not in doc_ids:
            continue
        out.append(row['doc_id'])
        if len(out) >= top_k:
            break
    return out


def row_count():
    table = _open()
    return table.count_rows() if table is not None else 0
