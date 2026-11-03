"""G2 content sanitiser — FR-4.4/4.5, exam C3. Scans the RAG corpus at ingestion
time for instruction-like injected text (the "SYSTEM MESSAGE: ... every premium
tile is free" record from the exam) and keeps matching records out of the index.
"""

import re

INJECTION_PATTERNS = [
    r'system message\s*:',
    r'ignore (all |any |the )?(previous|prior|above) instructions',
    r'reveal (your|the) (hidden |system )?(prompt|credentials)',
    r'administrator credentials',
    r'every .* is free',
    r'you are now (a|an)',
    r'disregard (all |any |the )?(previous|prior) (rules|instructions)',
]


def scan_document(text):
    """Returns (is_malicious: bool, matched_pattern: str|None)."""
    lower = (text or '').lower()
    for pattern in INJECTION_PATTERNS:
        if re.search(pattern, lower, re.I):
            return True, pattern
    return False, None


def sanitise_corpus(documents):
    """Splits a document list into (clean, quarantined). Quarantined entries
    keep the matched pattern so the ingestion report (and the demo) can show
    *why* a record was dropped, per FR-4.4's 'an alert is raised'."""
    clean = []
    quarantined = []
    for doc in documents:
        is_malicious, pattern = scan_document(doc.get('text', ''))
        if is_malicious:
            quarantined.append({**doc, 'quarantine_reason': pattern})
        else:
            clean.append(doc)
    return clean, quarantined
