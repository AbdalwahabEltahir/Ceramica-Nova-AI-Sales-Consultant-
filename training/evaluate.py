"""E.12: baseline vs domain-adapted model on the same held-out eval set,
same metrics, same decoding. The baseline is the *same loaded weights* with
the LoRA adapter disabled, so the only difference between the two arms is
the adapter.

Also scores the reference targets as a third arm: they must score 100%, which
shows the checker accepts correct answers and isn't the bottleneck.

Usage:
  python -m training.evaluate             # full eval set
  python -m training.evaluate --limit 5   # quick check
"""

import argparse
import json
import math
import os
import sys
import time
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch  # noqa: E402
from peft import PeftModel  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig  # noqa: E402

from training.common import ADAPTER_DIR, BASE_MODEL, EVAL_PATH, RESULTS_DIR, SEED, build_messages  # noqa: E402
from training.metrics import METRICS, score  # noqa: E402

GENERATION = dict(do_sample=False, max_new_tokens=320, repetition_penalty=1.0)
ARMS = ['baseline', 'adapted']


def load_jsonl(path):
    with open(path, encoding='utf-8') as fh:
        return [json.loads(line) for line in fh]


@torch.no_grad()
def generate(model, tokenizer, example, config):
    prompt = tokenizer.apply_chat_template(build_messages(example, include_answer=False),
                                           tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors='pt', add_special_tokens=False)
    start = time.time()
    out = model.generate(**inputs, generation_config=config)
    latency = time.time() - start
    new_tokens = out[0, inputs['input_ids'].shape[1]:]
    text = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
    return text, round(latency, 2), int(new_tokens.shape[0])


def sign_test_p(improved, worsened):
    """Exact two-sided binomial sign test on the discordant pairs."""
    n = improved + worsened
    if n == 0:
        return 1.0
    k = min(improved, worsened)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def pct(x):
    return f'{100 * x:.1f}%'


def summarise(rows):
    arms = ARMS + ['reference']
    summary = {'n': len(rows), 'metrics': {}, 'categories': {}, 'latency_s': {}, 'output_tokens': {}}
    for arm in arms:
        summary['metrics'][arm] = {m: sum(r['scores'][arm][m] for r in rows) / len(rows)
                                   for m in METRICS + ['strict_pass']}
    for arm in ARMS:
        summary['latency_s'][arm] = round(sum(r['latency'][arm] for r in rows) / len(rows), 2)
        summary['output_tokens'][arm] = round(sum(r['tokens'][arm] for r in rows) / len(rows), 1)
    by_cat = defaultdict(list)
    for r in rows:
        by_cat[r['category']].append(r)
    for cat, items in sorted(by_cat.items()):
        summary['categories'][cat] = {'n': len(items), **{
            arm: sum(i['scores'][arm]['strict_pass'] for i in items) / len(items) for arm in ARMS}}
    improved = [r for r in rows if r['scores']['adapted']['strict_pass'] and not r['scores']['baseline']['strict_pass']]
    worsened = [r for r in rows if r['scores']['baseline']['strict_pass'] and not r['scores']['adapted']['strict_pass']]
    summary['paired'] = {
        'improved': len(improved), 'worsened': len(worsened),
        'both_pass': sum(r['scores']['adapted']['strict_pass'] and r['scores']['baseline']['strict_pass'] for r in rows),
        'both_fail': sum(not r['scores']['adapted']['strict_pass'] and not r['scores']['baseline']['strict_pass'] for r in rows),
        'sign_test_p': sign_test_p(len(improved), len(worsened)),
    }
    return summary, improved, worsened


def _failed(scores):
    return ', '.join(m for m in METRICS if not scores[m]) or 'none'


def _example_block(r):
    return (f"**{r['id']}** ({r['category']}, {r['language']})\n\n"
            f"Customer: {r['message']}\n\n"
            f"Baseline (failed: {_failed(r['scores']['baseline'])}):\n```\n{r['outputs']['baseline']}\n```\n"
            f"Adapted (failed: {_failed(r['scores']['adapted'])}):\n```\n{r['outputs']['adapted']}\n```\n")


def write_report(rows, summary, improved, worsened, path):
    s = summary
    lines = ['# E.12 Baseline vs domain-adapted model', '',
             f"Base model `{BASE_MODEL}`; adapted = same weights + LoRA adapter (`training/outputs/adapter`).",
             f"Held-out eval set: {s['n']} examples (`training/data/eval.jsonl`), phrasings never seen in training.",
             f"Decoding (both arms): `{GENERATION}`. Seed {SEED}. Scoring: `training/metrics.py` (deterministic, no LLM judge).",
             '', '## Overall', '',
             '| Metric | Baseline | Adapted | Reference targets |', '|---|---|---|---|']
    for m in METRICS + ['strict_pass']:
        lines.append(f"| {m} | {pct(s['metrics']['baseline'][m])} | {pct(s['metrics']['adapted'][m])} | "
                     f"{pct(s['metrics']['reference'][m])} |")
    lines += ['', f"Mean generation latency (CPU): baseline {s['latency_s']['baseline']}s, adapted {s['latency_s']['adapted']}s; "
              f"mean output tokens: baseline {s['output_tokens']['baseline']}, adapted {s['output_tokens']['adapted']}.",
              '', '## Strict pass rate by category', '', '| Category | n | Baseline | Adapted |', '|---|---|---|---|']
    for cat, c in s['categories'].items():
        lines.append(f"| {cat} | {c['n']} | {pct(c['baseline'])} | {pct(c['adapted'])} |")
    p = s['paired']
    lines += ['', '## Paired comparison (strict pass)', '',
              f"- Adapted passes, baseline fails (improved): {p['improved']}",
              f"- Baseline passes, adapted fails (worsened): {p['worsened']}",
              f"- Both pass: {p['both_pass']}; both fail: {p['both_fail']}",
              f"- Exact two-sided sign test on discordant pairs: p = {p['sign_test_p']:.2g}", '']

    lines += ['## Examples where adaptation improved the result', '',
              '_Selection rule: first 4 improved examples in id order, no hand-picking._', '']
    lines += [_example_block(r) for r in improved[:4]] or ['None.']
    lines += ['## Examples where adaptation made results worse', '', '_All regressions are listed._', '']
    lines += [_example_block(r) for r in worsened] or ['None: no example passed on the baseline and failed after adaptation.', '']
    still_failing = [r for r in rows if not r['scores']['adapted']['strict_pass']]
    lines += ['## Examples where the adapted model still fails', '',
              '_Selection rule: first 4 adapted failures in id order._', '']
    lines += [_example_block(r) for r in still_failing[:4]] or ['None.']
    with open(path, 'w', encoding='utf-8') as fh:
        fh.write('\n'.join(lines) + '\n')


def rescore():
    """Re-scores the saved model outputs with the current metrics, without
    regenerating (generation is the slow part). Used when a metric is added
    or fixed after the outputs were read: the outputs stay exactly as the
    models produced them."""
    saved = {r['id']: r for r in load_jsonl(os.path.join(RESULTS_DIR, 'eval_outputs.jsonl'))}
    rows = []
    for ex in load_jsonl(EVAL_PATH):
        row = saved[ex['id']]
        row['scores'] = {arm: score(row['outputs'][arm], ex['checks']) for arm in ARMS + ['reference']}
        rows.append(row)
    with open(os.path.join(RESULTS_DIR, 'eval_outputs.jsonl'), 'w', encoding='utf-8') as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + '\n')
    summary, improved, worsened = summarise(rows)
    summary['generation'] = GENERATION
    with open(os.path.join(RESULTS_DIR, 'metrics.json'), 'w', encoding='utf-8') as fh:
        json.dump(summary, fh, indent=2)
    write_report(rows, summary, improved, worsened, os.path.join(RESULTS_DIR, 'comparison.md'))
    print(json.dumps({k: summary[k] for k in ('metrics', 'paired')}, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--limit', type=int, default=None)
    parser.add_argument('--rescore', action='store_true', help='re-score saved outputs, no generation')
    args = parser.parse_args()
    if args.rescore:
        rescore()
        return

    torch.manual_seed(SEED)
    rows_in = load_jsonl(EVAL_PATH)[:args.limit]
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    base = AutoModelForCausalLM.from_pretrained(BASE_MODEL, dtype=torch.float32)
    model = PeftModel.from_pretrained(base, ADAPTER_DIR)
    model.eval()
    config = GenerationConfig(**GENERATION, pad_token_id=tokenizer.eos_token_id,
                              eos_token_id=base.generation_config.eos_token_id)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    rows = []
    for i, ex in enumerate(rows_in, 1):
        row = {'id': ex['id'], 'category': ex['category'], 'language': ex['language'], 'message': ex['message'],
               'outputs': {}, 'latency': {}, 'tokens': {}, 'scores': {}}
        with model.disable_adapter():
            row['outputs']['baseline'], row['latency']['baseline'], row['tokens']['baseline'] = generate(
                model, tokenizer, ex, config)
        row['outputs']['adapted'], row['latency']['adapted'], row['tokens']['adapted'] = generate(
            model, tokenizer, ex, config)
        row['outputs']['reference'] = ex['target']
        for arm in ARMS + ['reference']:
            row['scores'][arm] = score(row['outputs'][arm], ex['checks'])
        rows.append(row)
        print(f"[{i}/{len(rows_in)}] {ex['id']}: baseline={row['scores']['baseline']['strict_pass']} "
              f"adapted={row['scores']['adapted']['strict_pass']}", flush=True)

    suffix = f'_limit{args.limit}' if args.limit else ''
    with open(os.path.join(RESULTS_DIR, f'eval_outputs{suffix}.jsonl'), 'w', encoding='utf-8') as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + '\n')
    summary, improved, worsened = summarise(rows)
    summary['generation'] = GENERATION
    with open(os.path.join(RESULTS_DIR, f'metrics{suffix}.json'), 'w', encoding='utf-8') as fh:
        json.dump(summary, fh, indent=2)
    write_report(rows, summary, improved, worsened, os.path.join(RESULTS_DIR, f'comparison{suffix}.md'))
    print(json.dumps({k: summary[k] for k in ('metrics', 'paired', 'latency_s')}, indent=2))


if __name__ == '__main__':
    main()
