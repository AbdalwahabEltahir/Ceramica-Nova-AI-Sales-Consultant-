"""Hugging Face Inference API clients for the three classification/
translation tasks that sit in front of the main Gemini conversation:
prompt-injection classification, sentiment analysis, and Arabic→English
translation (used only to help English-biased internal components — BM25
retrieval, off-topic keyword matching — not for anything customer-facing or
for FR-1.2 evidence extraction, which must stay traceable to the customer's
*original* words).

Same reliability posture as rag/embeddings_client.py: every function fails
soft (returns None) when HUGGINGFACE_API_KEY is unset, the model is cold-
starting, or the call errors — callers always have a non-ML fallback.
"""

import os
import time

import requests

# api-inference.huggingface.co no longer resolves (confirmed via live DNS
# lookup) — HF migrated serverless inference behind a routed, provider-based
# gateway. hf-inference is the provider name for HF's own hosted models.
API_URL_TEMPLATE = 'https://router.huggingface.co/hf-inference/models/{model}'
TIMEOUT_SECONDS = 12
MAX_RETRIES = 1

INJECTION_MODEL = os.environ.get('HF_INJECTION_MODEL', 'protectai/deberta-v3-base-prompt-injection-v2')
SENTIMENT_MODEL = os.environ.get('HF_SENTIMENT_MODEL', 'cardiffnlp/twitter-xlm-roberta-base-sentiment')
TRANSLATION_MODEL = os.environ.get('HF_TRANSLATION_MODEL', 'Helsinki-NLP/opus-mt-ar-en')

INJECTION_SCORE_THRESHOLD = 0.85
NEGATIVE_SENTIMENT_THRESHOLD = 0.6


def _api_key():
    return os.environ.get('HUGGINGFACE_API_KEY') or os.environ.get('HF_API_KEY')


def is_configured():
    return bool(_api_key())


def _post(model, payload):
    api_key = _api_key()
    if not api_key:
        return None
    url = API_URL_TEMPLATE.format(model=model)
    headers = {'Authorization': f'Bearer {api_key}'}
    for attempt in range(MAX_RETRIES + 1):
        try:
            response = requests.post(
                url, headers=headers,
                json={**payload, 'options': {'wait_for_model': True}},
                timeout=TIMEOUT_SECONDS,
            )
            if response.status_code == 200:
                return response.json()
            if response.status_code in (429, 503) and attempt < MAX_RETRIES:
                time.sleep(1.0)
                continue
            return None
        except requests.RequestException:
            if attempt < MAX_RETRIES:
                time.sleep(0.5)
                continue
            return None
    return None


def _flatten_classification(result):
    """HF text-classification responses are sometimes [[{...}, {...}]] and
    sometimes [{...}, {...}] depending on the model/pipeline version."""
    if not result:
        return []
    if isinstance(result[0], list):
        return result[0]
    return result


def classify_prompt_injection(text):
    """Returns (flagged: bool, label: str|None, score: float).
    ML layer in the G1 ensemble — regex patterns catch known phrasing fast
    and for free; this catches paraphrased/obfuscated attempts regex misses."""
    result = _post(INJECTION_MODEL, {'inputs': text})
    scores = _flatten_classification(result)
    if not scores:
        return False, None, 0.0
    best = max(scores, key=lambda s: s.get('score', 0))
    label = (best.get('label') or '').upper()
    score = best.get('score', 0.0)
    is_injection_label = label in ('INJECTION', 'JAILBREAK', 'LABEL_1', 'UNSAFE')
    return (is_injection_label and score >= INJECTION_SCORE_THRESHOLD), label, score


def analyze_sentiment(text):
    """Returns (label: 'positive'|'neutral'|'negative'|None, score: float).
    Feeds the price-sensitivity/objection-handling signals (E.14) — a
    ranking/tone input only, never a hard gate on anything."""
    result = _post(SENTIMENT_MODEL, {'inputs': text})
    scores = _flatten_classification(result)
    if not scores:
        return None, 0.0
    best = max(scores, key=lambda s: s.get('score', 0))
    return (best.get('label') or '').lower(), best.get('score', 0.0)


def translate_arabic_to_english(text):
    """Returns the English translation, or None on any failure. Used only to
    widen internal English-biased matching (BM25 query terms, off-topic
    vocabulary check) — never shown to the customer and never fed into FR-1
    evidence extraction, which must quote the customer's own words."""
    result = _post(TRANSLATION_MODEL, {'inputs': text})
    if not result or not isinstance(result, list):
        return None
    first = result[0]
    if isinstance(first, dict):
        return first.get('translation_text')
    return None
