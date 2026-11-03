"""Dense embeddings for RAG (FR-3.2). Two hosted backends, both multilingual
so an Arabic query can match English product text:

  gemini — gemini-embedding-001 via GOOGLE_API_KEY (default when that key is set)
  hf     — sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 via
           HUGGINGFACE_API_KEY

Pick explicitly with EMBEDDING_PROVIDER=gemini|hf. Documents and queries must
be embedded by the same model, so model_id() is stored alongside every vector
(rag/index.py cache, rag/lancedb_store.py table) and a model change forces a
re-embed. Every call fails soft (returns None) so the retriever degrades to
BM25-only instead of failing the turn.
"""

import logging
import os
import time

import requests

logger = logging.getLogger('embeddings')

HF_DEFAULT_MODEL = 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'
HF_API_URL_TEMPLATE = 'https://router.huggingface.co/hf-inference/models/{model}'
GEMINI_DEFAULT_MODEL = 'gemini-embedding-001'
GEMINI_DIMENSIONS = 768
GEMINI_BATCH = 100
TIMEOUT_SECONDS = 15
MAX_RETRIES = 2


def _hf_key():
    return os.environ.get('HUGGINGFACE_API_KEY') or os.environ.get('HF_API_KEY')


def _provider():
    choice = (os.environ.get('EMBEDDING_PROVIDER') or '').lower()
    if choice in ('gemini', 'hf'):
        return choice
    if os.environ.get('GOOGLE_API_KEY'):
        return 'gemini'
    return 'hf'


def is_configured():
    if _provider() == 'gemini':
        return bool(os.environ.get('GOOGLE_API_KEY'))
    return bool(_hf_key())


def model_id():
    if _provider() == 'gemini':
        return f"gemini:{os.environ.get('GEMINI_EMBEDDING_MODEL', GEMINI_DEFAULT_MODEL)}:{GEMINI_DIMENSIONS}"
    return f"hf:{os.environ.get('HF_EMBEDDING_MODEL', HF_DEFAULT_MODEL)}"


def _mean_pool(vectors):
    # Some HF models return per-token embeddings; mean-pool to one vector.
    if not vectors:
        return None
    if isinstance(vectors[0], (int, float)):
        return vectors
    n = len(vectors)
    dims = len(vectors[0])
    return [sum(v[d] for v in vectors) / n for d in range(dims)]


def _embed_hf(texts):
    url = HF_API_URL_TEMPLATE.format(model=os.environ.get('HF_EMBEDDING_MODEL', HF_DEFAULT_MODEL))
    headers = {'Authorization': f'Bearer {_hf_key()}'}
    for attempt in range(MAX_RETRIES + 1):
        try:
            response = requests.post(
                url, headers=headers,
                json={'inputs': texts, 'options': {'wait_for_model': True}},
                timeout=TIMEOUT_SECONDS,
            )
            if response.status_code == 200:
                return [_mean_pool(item) for item in response.json()]
            if response.status_code in (429, 503) and attempt < MAX_RETRIES:
                time.sleep(1.5 * (attempt + 1))
                continue
            logger.info('HF embeddings returned %s', response.status_code)
            return [None] * len(texts)
        except requests.RequestException:
            if attempt < MAX_RETRIES:
                time.sleep(1.0 * (attempt + 1))
                continue
            return [None] * len(texts)
    return [None] * len(texts)


def _embed_gemini_batch(batch, task_type, model):
    """One batch, trying each configured key in order (a second key from a
    separate project has its own quota). Returns vectors or None."""
    from google.genai import types
    from backend.app.services.llm.gemini_provider import _get_client, api_keys

    last_error = None
    for api_key in api_keys():
        for attempt in range(MAX_RETRIES + 1):
            try:
                response = _get_client(api_key).models.embed_content(
                    model=model, contents=batch,
                    config=types.EmbedContentConfig(task_type=task_type,
                                                    output_dimensionality=GEMINI_DIMENSIONS),
                )
                return [list(e.values) for e in response.embeddings]
            except Exception as exc:  # network/quota — fail soft, never surface internals
                last_error = exc
                if '429' in str(exc) or 'RESOURCE_EXHAUSTED' in str(exc):
                    break  # quota is per key: retrying this key is pointless, try the next
                if attempt < MAX_RETRIES:
                    time.sleep(1.5 * (attempt + 1))
    # Warning, not info: a silent fall back to BM25-only once hid a real bug.
    logger.warning('Gemini embeddings failed on every key, retrieval is BM25-only: %s', last_error)
    return None


def _embed_gemini(texts, task_type):
    model = os.environ.get('GEMINI_EMBEDDING_MODEL', GEMINI_DEFAULT_MODEL)
    out = []
    for start in range(0, len(texts), GEMINI_BATCH):
        batch = texts[start:start + GEMINI_BATCH]
        vectors = _embed_gemini_batch(batch, task_type, model)
        out.extend(vectors if vectors is not None else [None] * len(batch))
    return out


def embed_texts(texts, task='document'):
    """Returns a list of embedding vectors (None entries on failure).
    task: 'document' when indexing, 'query' when searching."""
    if not texts or not is_configured():
        return [None] * len(texts or [])
    if _provider() == 'gemini':
        return _embed_gemini(texts, 'RETRIEVAL_QUERY' if task == 'query' else 'RETRIEVAL_DOCUMENT')
    return _embed_hf(texts)


def embed_text(text, task='query'):
    return embed_texts([text], task=task)[0]
