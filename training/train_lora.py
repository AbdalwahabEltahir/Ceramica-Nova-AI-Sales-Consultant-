"""E.11: LoRA supervised fine-tuning of Qwen2.5-0.5B-Instruct on the
generated ceramics-sales dataset. CPU-only by design (no GPU on the dev
machine); the model is small enough that a full run fits in about an hour.

Loss is computed on the assistant reply tokens only: the system prompt and
CONTEXT are inputs the model should condition on, not text to imitate.

Usage:
  python -m training.train_lora                # full run
  python -m training.train_lora --max-steps 2  # smoke test / timing
"""

import argparse
import json
import math
import os
import platform
import random
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch  # noqa: E402
from peft import LoraConfig, get_peft_model  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

from training.common import (  # noqa: E402
    ADAPTER_DIR, BASE_MODEL, EVAL_PATH, LOG_DIR, SEED, TRAIN_PATH, build_messages)

HPARAMS = {
    'lora_r': 16,
    'lora_alpha': 32,
    'lora_dropout': 0.05,
    'target_modules': ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj'],
    'learning_rate': 2e-4,
    'epochs': 2,
    'grad_accum_steps': 8,  # effective batch size 8 (one sequence per forward pass, no padding)
    'warmup_steps': 6,
    'max_grad_norm': 1.0,
    'max_seq_len': 1024,
    'seed': SEED,
}


def load_jsonl(path):
    with open(path, encoding='utf-8') as fh:
        return [json.loads(line) for line in fh]


def encode(tokenizer, example):
    """input_ids for the full conversation, labels = -100 everywhere except
    the assistant reply (and its end-of-turn token)."""
    prompt = tokenizer.apply_chat_template(build_messages(example, include_answer=False),
                                           tokenize=False, add_generation_prompt=True)
    full = tokenizer.apply_chat_template(build_messages(example), tokenize=False)
    assert full.startswith(prompt), 'chat template is not prefix-stable'
    prompt_ids = tokenizer(prompt, add_special_tokens=False)['input_ids']
    full_ids = tokenizer(full, add_special_tokens=False)['input_ids']
    if len(full_ids) > HPARAMS['max_seq_len']:
        raise ValueError(f"{example['id']} is {len(full_ids)} tokens, above max_seq_len")
    labels = [-100] * len(prompt_ids) + full_ids[len(prompt_ids):]
    return torch.tensor([full_ids]), torch.tensor([labels])


def lr_at(step, total):
    if step < HPARAMS['warmup_steps']:
        return HPARAMS['learning_rate'] * (step + 1) / HPARAMS['warmup_steps']
    progress = (step - HPARAMS['warmup_steps']) / max(1, total - HPARAMS['warmup_steps'])
    return HPARAMS['learning_rate'] * max(0.0, 1.0 - progress)  # linear decay to 0


@torch.no_grad()
def eval_loss(model, encoded):
    model.eval()
    total_loss, total_tokens = 0.0, 0
    for input_ids, labels in encoded:
        out = model(input_ids=input_ids, labels=labels)
        n = int((labels[:, 1:] != -100).sum())
        total_loss += out.loss.item() * n
        total_tokens += n
    model.train()
    return total_loss / total_tokens


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--max-steps', type=int, default=None, help='stop after N optimizer steps (smoke test)')
    args = parser.parse_args()

    random.seed(SEED)
    torch.manual_seed(SEED)
    os.makedirs(LOG_DIR, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, dtype=torch.float32)
    model.config.use_cache = False
    model = get_peft_model(model, LoraConfig(
        r=HPARAMS['lora_r'], lora_alpha=HPARAMS['lora_alpha'], lora_dropout=HPARAMS['lora_dropout'],
        target_modules=HPARAMS['target_modules'], task_type='CAUSAL_LM'))
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f'trainable params: {trainable:,} / {total:,} ({100 * trainable / total:.2f}%)')

    train_rows, eval_rows = load_jsonl(TRAIN_PATH), load_jsonl(EVAL_PATH)
    train_enc = [encode(tokenizer, r) for r in train_rows]
    eval_enc = [encode(tokenizer, r) for r in eval_rows]
    lengths = [ids.shape[1] for ids, _ in train_enc]
    print(f'train sequences: {len(train_enc)} | tokens min/mean/max: '
          f'{min(lengths)}/{sum(lengths) / len(lengths):.0f}/{max(lengths)}')

    steps_per_epoch = math.ceil(len(train_enc) / HPARAMS['grad_accum_steps'])
    total_steps = steps_per_epoch * HPARAMS['epochs']
    if args.max_steps:
        total_steps = min(total_steps, args.max_steps)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                  lr=HPARAMS['learning_rate'], weight_decay=0.0)

    log_path = os.path.join(LOG_DIR, 'train_log.jsonl')
    log = open(log_path, 'w', encoding='utf-8')
    started = time.time()
    baseline_eval_loss = eval_loss(model, eval_enc) if not args.max_steps else None
    if baseline_eval_loss is not None:
        print(f'eval loss before training: {baseline_eval_loss:.4f}')
        log.write(json.dumps({'event': 'eval', 'epoch': 0, 'eval_loss': baseline_eval_loss}) + '\n')

    step, epoch_losses = 0, []
    model.train()
    for epoch in range(1, HPARAMS['epochs'] + 1):
        order = list(range(len(train_enc)))
        random.shuffle(order)
        for batch_start in range(0, len(order), HPARAMS['grad_accum_steps']):
            if step >= total_steps:
                break
            batch = order[batch_start:batch_start + HPARAMS['grad_accum_steps']]
            batch_loss = 0.0
            for idx in batch:
                input_ids, labels = train_enc[idx]
                loss = model(input_ids=input_ids, labels=labels).loss / len(batch)
                loss.backward()
                batch_loss += loss.item()
            grad_norm = torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], HPARAMS['max_grad_norm'])
            lr = lr_at(step, total_steps)
            for group in optimizer.param_groups:
                group['lr'] = lr
            optimizer.step()
            optimizer.zero_grad()
            step += 1
            epoch_losses.append(batch_loss)
            record = {'event': 'step', 'step': step, 'epoch': epoch, 'loss': round(batch_loss, 4),
                      'lr': lr, 'grad_norm': round(float(grad_norm), 4),
                      'elapsed_s': round(time.time() - started, 1)}
            log.write(json.dumps(record) + '\n')
            log.flush()
            print(f"step {step}/{total_steps} epoch {epoch} loss {batch_loss:.4f} lr {lr:.2e} "
                  f"elapsed {record['elapsed_s']}s", flush=True)
        if args.max_steps and step >= total_steps:
            break
        ev = eval_loss(model, eval_enc)
        print(f'epoch {epoch} done | eval loss {ev:.4f}', flush=True)
        log.write(json.dumps({'event': 'eval', 'epoch': epoch, 'eval_loss': ev,
                              'elapsed_s': round(time.time() - started, 1)}) + '\n')
        log.flush()
    log.close()

    duration = time.time() - started
    if args.max_steps:
        print(f'smoke test: {step} steps in {duration:.1f}s ({duration / step:.1f}s/step)')
        return

    model.save_pretrained(ADAPTER_DIR)
    summary = {
        'base_model': BASE_MODEL,
        'method': 'LoRA supervised fine-tuning (PEFT), loss on assistant tokens only',
        'hyperparameters': HPARAMS,
        'trainable_params': trainable,
        'total_params': total,
        'train_examples': len(train_enc),
        'eval_examples': len(eval_enc),
        'optimizer_steps': step,
        'train_tokens_mean': round(sum(lengths) / len(lengths)),
        'train_tokens_max': max(lengths),
        'eval_loss_before': baseline_eval_loss,
        'duration_minutes': round(duration / 60, 1),
        'hardware': {'device': 'cpu', 'processor': platform.processor(), 'torch_threads': torch.get_num_threads()},
        'versions': {'torch': torch.__version__, 'python': platform.python_version()},
    }
    with open(LOG_DIR + '/train_log.jsonl', encoding='utf-8') as fh:
        evals = [json.loads(l) for l in fh if '"eval"' in l]
    summary['eval_loss_by_epoch'] = {e['epoch']: round(e['eval_loss'], 4) for e in evals}
    with open(os.path.join(LOG_DIR, 'train_summary.json'), 'w', encoding='utf-8') as fh:
        json.dump(summary, fh, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
