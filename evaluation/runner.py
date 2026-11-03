"""FR-10 evaluation runner. Drives every case in evaluation/cases/*.yaml through
/api/v1/chat (in-process test client, no network) and writes a report.

Usage: python -m evaluation.runner
"""

import datetime
import glob
import os
import sys

import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import app  # noqa: E402

CASES_DIR = os.path.join(os.path.dirname(__file__), 'cases')
REPORTS_DIR = os.path.join(os.path.dirname(__file__), 'reports')


def load_cases():
    cases = []
    for path in sorted(glob.glob(os.path.join(CASES_DIR, '*.yaml'))):
        with open(path, 'r', encoding='utf-8') as fh:
            cases.append(yaml.safe_load(fh))
    return cases


def run_case(case):
    client = app.test_client()  # fresh session per case — cases must not leak state
    payload = None
    for turn in case['turns']:
        response = client.post('/api/v1/chat', json={'message': turn})
        payload = response.get_json()

    expect = case.get('expect', {})
    failures = []

    for field, value in expect.get('requirements', {}).items():
        actual = payload['requirements'].get(field)
        if actual != value:
            failures.append(f"requirements.{field} expected {value!r}, got {actual!r}")

    skus = {r['sku'] for r in payload['recommendations']}
    any_of = set(expect.get('products_any_of', []))
    if any_of and not (skus & any_of):
        failures.append(f'expected one of {any_of}, got {skus}')
    none_of = set(expect.get('products_none_of', []))
    if skus & none_of:
        failures.append(f'forbidden products present: {skus & none_of}')

    max_products = expect.get('max_products')
    if max_products is not None and len(payload['recommendations']) > max(max_products, 3):
        failures.append(f'too many products: {len(payload["recommendations"])}')

    for key, expected_value in expect.get('numbers', {}).items():
        sku, field = key.split('.')
        rec = next((r for r in payload['recommendations'] if r['sku'] == sku), None)
        if not rec:
            failures.append(f'{sku} not in recommendations, cannot check {field}')
        elif abs(rec.get(field, 0) - expected_value) > 0.01:
            failures.append(f'{key} expected {expected_value}, got {rec.get(field)}')

    reply_lower = payload['response'].lower()
    for phrase in expect.get('forbidden_phrases', []):
        if phrase.lower() in reply_lower:
            failures.append(f'forbidden phrase present: {phrase!r}')
    for phrase in expect.get('required_phrases', []):
        if phrase.lower() not in reply_lower:
            failures.append(f'required phrase missing: {phrase!r}')

    if 'min_clarifying_questions' in expect:
        if len(payload['clarifying_questions']) < expect['min_clarifying_questions']:
            failures.append('fewer clarifying questions than expected')
    if 'max_clarifying_questions' in expect:
        if len(payload['clarifying_questions']) > expect['max_clarifying_questions']:
            failures.append('more clarifying questions than the BR-10 cap of 2')

    if 'guardrail_triggered' in expect:
        actual = payload.get('guardrail', {}).get('triggered', False)
        if actual != expect['guardrail_triggered']:
            failures.append(f'guardrail_triggered expected {expect["guardrail_triggered"]}, got {actual}')

    return {'id': case['id'], 'category': case['category'], 'passed': len(failures) == 0, 'failures': failures}


def main():
    os.makedirs(REPORTS_DIR, exist_ok=True)
    results = [run_case(case) for case in load_cases()]

    passed = sum(1 for r in results if r['passed'])
    total = len(results)
    run_id = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')

    lines = [f'# Evaluation run {run_id}', '', f'Pass rate: {passed}/{total}', '', '| ID | Category | Status | Failures |', '|---|---|---|---|']
    for r in results:
        status = 'PASS' if r['passed'] else 'FAIL'
        lines.append(f"| {r['id']} | {r['category']} | {status} | {'; '.join(r['failures']) or '-'} |")

    report_path = os.path.join(REPORTS_DIR, f'{run_id}.md')
    with open(report_path, 'w', encoding='utf-8') as fh:
        fh.write('\n'.join(lines))

    print('\n'.join(lines))
    print(f'\nReport saved to {report_path}')
    return 0 if passed == total else 1


if __name__ == '__main__':
    raise SystemExit(main())
