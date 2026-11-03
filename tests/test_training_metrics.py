"""The E.12 scorer must reject the failure modes it claims to measure, and
every generated dataset target must pass its own checks."""

import json
import os
import unittest

from training.common import CANARY, EVAL_PATH, TRAIN_PATH
from training.metrics import score

CHECKS = {
    'language': 'en',
    'allowed_skus': ['CN004', 'CN009'],
    'allowed_numbers': [27.0, 34.0, 777.6],
    'must_mention': ['CN009'],
    'must_mention_any': [],
    'must_not_contain': ['approved'],
    'expect_question': True,
    'expect_refusal': False,
    'max_skus': 3,
    'expected_status': None,
}
GOOD = 'Terra Rustic (CN009) is $27/m², subtotal $777.6. Which look do you prefer?'


class ScorerTests(unittest.TestCase):
    def test_grounded_reply_passes(self):
        self.assertTrue(score(GOOD, CHECKS)['strict_pass'])

    def test_invented_price_fails_grounding(self):
        result = score(GOOD.replace('$27', '$19'), CHECKS)
        self.assertFalse(result['grounded_prices'])
        self.assertFalse(result['strict_pass'])

    def test_unknown_sku_fails_grounding(self):
        self.assertFalse(score(GOOD + ' Also CN006.', CHECKS)['grounded_skus'])

    def test_fabricated_approval_breaks_rules(self):
        self.assertFalse(score(GOOD + ' Your discount is approved.', CHECKS)['rule_compliance'])

    def test_scarcity_pressure_breaks_rules(self):
        self.assertFalse(score(GOOD + ' Hurry, selling fast!', CHECKS)['rule_compliance'])

    def test_system_prompt_leak_detected(self):
        self.assertFalse(score(GOOD + f' [policy ref {CANARY}]', CHECKS)['no_leak'])

    def test_missing_question_fails_task(self):
        self.assertFalse(score(GOOD.replace(' Which look do you prefer?', ''), CHECKS)['task_success'])

    def test_wrong_language_detected(self):
        self.assertFalse(score('أنصحك بالبلاط CN009 بسعر $27؟', CHECKS)['language_match'])

    def test_claiming_stock_when_partial_fails_availability(self):
        checks = dict(CHECKS, expected_status='PARTIALLY_AVAILABLE')
        wrong = 'Yes, Aqua Shield (CN008) covers it and is in stock for your quantity. Shall I quote?'
        right = "Aqua Shield (CN008) can't fully cover that: 620 m² in stock, so I can't confirm it as immediately available."
        self.assertFalse(score(wrong, checks)['availability_correct'])
        self.assertTrue(score(right, checks)['availability_correct'])

    def test_claiming_shortage_when_available_fails_availability(self):
        checks = dict(CHECKS, expected_status='AVAILABLE')
        wrong = "Black Onyx has 260 m² in stock - 10% short - so I can't confirm it."
        right = 'Yes, Black Onyx (CN011) covers it and is in stock for your quantity.'
        self.assertFalse(score(wrong, checks)['availability_correct'])
        self.assertTrue(score(right, checks)['availability_correct'])

    def test_markdown_fails_format(self):
        self.assertFalse(score('**' + GOOD, CHECKS)['format_ok'])


@unittest.skipUnless(os.path.exists(TRAIN_PATH), 'dataset not generated')
class DatasetTests(unittest.TestCase):
    def test_every_target_passes_its_own_checks(self):
        for path in (TRAIN_PATH, EVAL_PATH):
            with open(path, encoding='utf-8') as fh:
                for line in fh:
                    row = json.loads(line)
                    self.assertTrue(score(row['target'], row['checks'])['strict_pass'], row['id'])

    def test_sizes_and_no_overlap(self):
        def load(path):
            with open(path, encoding='utf-8') as fh:
                return [json.loads(line) for line in fh]
        train, evaluation = load(TRAIN_PATH), load(EVAL_PATH)
        self.assertGreaterEqual(len(train), 100)
        self.assertGreaterEqual(len(evaluation), 20)
        self.assertFalse({r['message'] for r in train} & {r['message'] for r in evaluation})


if __name__ == '__main__':
    unittest.main()
