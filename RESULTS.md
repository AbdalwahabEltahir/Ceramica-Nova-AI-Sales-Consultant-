# Results

All numbers below come from files in this repo and can be reproduced (commands at the end).

## 1. Baseline vs domain-adapted model

Base model `Qwen/Qwen2.5-0.5B-Instruct`. Adapted = the same weights plus a LoRA adapter
(`training/outputs/adapter`, r=16, 8.8 M of 503 M parameters trained, 260 synthetic training
examples across 9 behaviours, about 32 minutes on CPU). Both arms use the same decoding settings
and seed on 50 held-out prompts that never appear in training. Scoring is deterministic
(`training/metrics.py`, no LLM judge).

| Metric | Baseline | Adapted |
|---|---|---|
| Strict pass (all checks) | 16% | **78%** |
| Task success | 28% | **80%** |
| Grounded prices | 86% | **100%** |
| Grounded SKUs | 100% | 100% |
| Rule compliance | 96% | 98% |
| Format OK | 84% | **100%** |
| No leakage | 98% | **100%** |
| Language match | 96% | **100%** |
| Availability correct | 92% | 92% |

Paired comparison: 31 prompts improved, 0 got worse, exact sign-test p = 9.3e-10.
Mean CPU latency rose from 7.7 s to 11.2 s. Full tables: `reports/baseline-vs-adapted.md`,
raw numbers: `training/results/metrics.json`.

## 2. End-to-end scenario evaluation

`python -m evaluation.runner` runs 8 YAML cases through the real Flask app and pipeline:
recommendation (x2), wrong-recommendation trap, missing information, inventory, two prompt
injections, and one Arabic case. Latest run: **8/8 pass** (`reports/evaluation-latest.md`).

## 3. Unit and integration tests

`python -m pytest` runs 78 tests covering the API, business rules, calculator (including a
float-trap case), inventory, guardrails, RAG, training metrics and the published SRS scenarios.
None of them make live LLM calls.

## Limitations of these results

- The adapter is evaluated offline. It is **not** wired into the live chat, which uses Gemini plus
  deterministic rules, with a rule-based fallback.
- The held-out set is small (50) and synthetic, with only 4 Arabic prompts. The comparison is
  adapter vs no adapter on a 0.5 B model, not adapter vs the production Gemini path.
- The inventory category is still weak (16.7% strict pass after adaptation).
- The scenario suite has 8 cases, not the full 36 in the SRS.

## Reproduce

```bash
python -m pip install -r requirements.txt
python -m pytest                       # unit + scenario tests
python -m evaluation.runner            # 8 end-to-end cases
python -m pip install -r training/requirements.txt
python -m training.evaluate            # baseline vs adapted (CPU, slow)
```
