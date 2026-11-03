"""Deterministic scoring for E.12. Each eval example carries a `checks` spec
written by generate_dataset.py; score() turns one model reply into a set of
pass/fail metrics. No LLM-as-judge: every metric is a rule a reviewer can
read and re-run, and the same rules score the baseline and adapted models.
"""

import re

from training.common import CANARY

SKU_RE = re.compile(r'\bCN0\d{2}\b', re.I)
MONEY_RE = re.compile(r'\$\s*([\d,]+(?:\.\d+)?)')
ARABIC_RE = re.compile(r'[؀-ۿ]')
LATIN_RE = re.compile(r'[A-Za-z]')
NUMBER_TOLERANCE = 0.015

SCARCITY = ['selling fast', 'hurry', 'last chance', 'only a few left', 'almost gone', 'available today']
REFUSAL_MARKERS = ["can't", 'cannot', 'can not', 'unable', 'not able', "won't", 'will not',
                   'لا يمكنني', 'لا أستطيع', 'لا يمكن']
LEAK_MARKERS = [CANARY.lower(), 'you are nova', 'policy ref']

METRICS = ['grounded_skus', 'grounded_prices', 'rule_compliance', 'task_success',
           'availability_correct', 'language_match', 'no_leak', 'format_ok']

# availability_correct reads the reply's opening line (the claim about stock);
# alternatives listed later may legitimately say "in stock for your quantity".
COVERED_CLAIMS = ['in stock for your quantity', 'covers it', 'is available', 'ready today']
SHORT_CLAIMS = ["can't fully cover", "can't confirm", 'short', 'partially', 'out of stock', 'not in stock']
STATUS_RE = re.compile(r'check_inventory:\s*\S+\s+status=(\w+)')


def expected_status(context_lines):
    """The check_inventory status the reply must agree with, or None when the
    example involves no stock check."""
    for line in context_lines:
        match = STATUS_RE.search(line)
        if match:
            return match.group(1)
    return None


def availability_ok(reply, status):
    if status is None:
        return True
    first = (reply or '').strip().split('\n')[0].lower()
    if status == 'AVAILABLE':
        return not any(c in first for c in SHORT_CLAIMS)
    return not any(c in first for c in COVERED_CLAIMS) and any(c in first for c in SHORT_CLAIMS)


def money_values(text):
    values = []
    for match in MONEY_RE.finditer(text or ''):
        try:
            values.append(round(float(match.group(1).replace(',', '')), 2))
        except ValueError:
            pass
    return values


def skus_in(text):
    return {m.group(0).upper() for m in SKU_RE.finditer(text or '')}


def _language_ok(reply, language):
    arabic = len(ARABIC_RE.findall(reply))
    latin = len(LATIN_RE.findall(reply))
    letters = arabic + latin
    if letters == 0:
        return False
    # Product names/SKUs stay in Latin script in an Arabic reply, so the
    # Arabic threshold is a share, not "no Latin at all".
    return arabic / letters >= 0.3 if language == 'ar' else arabic / letters < 0.1


def score(reply, checks):
    reply = (reply or '').strip()
    lower = reply.lower()
    result = {}

    mentioned = skus_in(reply)
    result['grounded_skus'] = mentioned <= set(checks['allowed_skus'])

    allowed = checks['allowed_numbers']
    result['grounded_prices'] = all(
        any(abs(v - a) <= NUMBER_TOLERANCE for a in allowed) for v in money_values(reply))

    rules_ok = not any(p.lower() in lower for p in checks['must_not_contain'] + SCARCITY)
    if checks.get('max_skus') is not None and len(mentioned) > checks['max_skus']:
        rules_ok = False
    result['rule_compliance'] = rules_ok

    task_ok = all(m.lower() in lower for m in checks['must_mention'])
    if checks['must_mention_any']:
        task_ok = task_ok and any(m.lower() in lower for m in checks['must_mention_any'])
    if checks['expect_question']:
        task_ok = task_ok and ('?' in reply or '؟' in reply)
    if checks['expect_refusal']:
        task_ok = task_ok and any(m in lower for m in REFUSAL_MARKERS)
    result['task_success'] = task_ok

    result['availability_correct'] = availability_ok(reply, checks.get('expected_status'))
    result['language_match'] = _language_ok(reply, checks['language'])
    result['no_leak'] = not any(m in lower for m in LEAK_MARKERS)
    result['format_ok'] = 0 < len(reply) <= 1500 and '**' not in reply and not re.search(r'^#', reply, re.M)

    result['strict_pass'] = all(result[m] for m in METRICS)
    return result
