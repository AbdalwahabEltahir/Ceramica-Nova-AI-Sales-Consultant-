# Ceramica Nova AI Sales Consultant — MVP

A Flask implementation of the Ceramica Nova AI Sales Consultant, built against the
MVP Software Requirements Specification. The LLM understands and explains;
deterministic code decides and calculates; guardrails enforce the business rules
in code, not only in a prompt.

## Stack

- Python 3.13, Flask 3.x
- Pydantic v2 for the `Requirements` schema
- Plain server-rendered HTML + vanilla JS (no build step, no framework)
- In-memory session and quote stores (swap for Redis/SQLite without touching callers)

## Run locally

```bash
python -m pip install -r requirements.txt
python app.py
```

Then open http://localhost:5000 in your browser.

## Run the tests

```bash
python -m unittest discover -s tests -v
```

## Run the evaluation suite (FR-10)

```bash
python -m evaluation.runner
```

Scores every case in `evaluation/cases/*.yaml` against a fresh session and writes
a report to `evaluation/reports/<run-id>.md`. Unlike the test suite, this *does*
use live Gemini/HF calls when `.env` has real keys — see the quota note below
before running it repeatedly.

## Live provider status (as verified against real keys)

- **Gemini**: confirmed working against a real `GOOGLE_API_KEY`. The `gemini-2.5-*`
  model names are deprecated for new API keys (confirmed live: 404 "no longer
  available to new users") — `router.py` now defaults to `gemini-3.5-flash` /
  `gemini-3.5-flash-lite` / `gemini-pro-latest`, all confirmed reachable.
  **Free tier is small**: 5 requests/minute and ~20 requests/day per model.
  Normal demo usage will hit it — this is expected, not a bug. When it does,
  `resilient_provider.py`'s retry → fallback chain kicks in automatically and
  the customer still gets a correct (template-composed) answer; nothing breaks.
- **Hugging Face**: the key itself authenticates correctly, but `api-inference.
  huggingface.co` no longer resolves at all (HF migrated to a routed gateway —
  fixed to `router.huggingface.co/hf-inference/...`). With that fixed, calls now
  reach the server but return **402 "no remaining credits"** — HF's Inference
  Providers system requires pre-paid credits or a PRO plan for most hosted
  models now; there's no code-level fix for this, it's an account/billing state.
  Retrieval, sentiment, and translation all fall back cleanly to their
  non-ML paths (BM25-only, no signal, no translation) when this happens.
- **Tests never call either provider live** — `tests/test_*.py` force-clear
  both API keys before importing the app, specifically so the suite stays
  fast, deterministic, and doesn't burn your daily Gemini quota on every run.

## What's implemented

- **FR-1 Requirement discovery** — rule-based extractor (English + Arabic), essential-info
  staging, session merge, sq ft → m² conversion, ≤2 targeted questions per turn (BR-10).
- **FR-2 Deterministic tools** — `calculate_tile_quantity`, `check_inventory`,
  `generate_quote`, all `Decimal`-based (no float arithmetic on money/area).
- **FR-3 Hard constraints, retrieval and ranking** — BR-01/02/03/04/08 enforced in
  `models/rules.py`, soft-ranking with configurable weights, max 3 products, named-product
  suitability checks (e.g. "Arctic Pearl for a pool deck" is explicitly rejected with reasons).
- **FR-4 Guardrails** — G1 input guard (role override / prompt extraction / false authority /
  cross-customer / false-facts patterns, EN+AR), G4 output guard (grounding, scarcity
  language, delivery-date promises, unauthorised approval claims), G3 tool whitelist.
- **FR-5 Orchestration + API** — session-scoped turn loop, `/api/v1/chat`,
  `/api/v1/tools/*`, `/api/v1/quotes*`, `/api/v1/products*`, `/health`, `/metrics`,
  uniform error model.
- **FR-6 Web UI** — chat log, clarifying-question chips, product cards with
  availability badges and reason codes, quantity summary, quote panel with a
  pending-approval badge, EN/AR toggle, `textContent`-only rendering (no `innerHTML`).
- **FR-9 Sales signals** — price sensitivity, urgency and purchase-readiness detection
  feed ranking and stage transitions.
- **FR-10 Evaluation suite** — YAML cases + automated runner (a representative subset,
  not the full 36-case suite).

## Known gaps vs. the full SRS (by design, for this pass)

- **LLM is optional.** With a `GOOGLE_API_KEY` in `.env`, Gemini composes replies (tiered by
  task in `services/llm/router.py`). With no key, or when the free-tier quota is exhausted,
  `services/composer.py` acts as the FR-7.3 *template fallback* and the app still answers
  correctly. The guardrails, tools and schemas don't depend on the provider.
- **RAG/vector index** (FR-3.2) is implemented: `rag/` builds a hybrid BM25 +
  dense index over the product/rule corpus. Dense vectors come from
  `rag/embeddings_client.py` (Gemini `gemini-embedding-001` by default, HF
  optional) and are stored with every catalogue column in a local LanceDB
  table (`rag/lancedb_store.py`, data in `rag/.lancedb/`, gitignored; rebuild
  with `python -m rag.ingest`). Retrieval fuses BM25 + LanceDB vector search
  via reciprocal rank fusion and degrades to BM25-only if no embedding key is
  set or the call fails.
- **Domain adaptation is offline only.** `training/` fine-tunes Qwen2.5-0.5B-Instruct with LoRA and
  compares it with its own baseline (strict pass 16% to 78%, see `RESULTS.md`). The adapter is
  not wired into the live chat.
- **Evaluation suite has 8 cases, not 36** — covers recommendation, a wrong-recommendation
  trap, missing-information, inventory, two injection attacks, and one Arabic case.
- **No staff auth** on `/api/v1/quotes/<id>/approve` — anyone can call it in this build;
  a real deployment needs the staff role check from the SRS user-classes table.

## Project layout

```
backend/app/
  config.py          business-rule values (wastage %, discount limit, weights)
  models/            Requirements schema, BR-01..BR-10, money helpers, catalogue, sessions
  services/
    discovery.py     FR-1 requirement extraction
    recommender.py   FR-3 hard constraints + ranking
    composer.py      template reply builder (FR-7.3 fallback)
    orchestrator.py  FR-5 turn loop
    guardrails/       G1 input, G3 tool, G4 output
    tools/            calculator, inventory, quotation, registry
  controllers/        one blueprint per resource
  views/errors.py     FR-5.4 error model
frontend/             templates/, static/css, static/js — plain HTML/CSS/JS
data/                 prompt-injection fixtures
rag/                  hybrid BM25 + LanceDB retrieval, ingest sanitiser
training/             LoRA dataset, training, baseline-vs-adapted evaluation
evaluation/           cases/*.yaml, runner.py
tests/                unit + scenario tests
security/             threat model and controls
architecture/         interactive diagram and flow description
reports/              evaluation report, baseline-vs-adapted tables, demo script
RESULTS.md            headline results and how to reproduce them
AI_USAGE.md           where AI is used, at runtime and in building
Dockerfile, docker-compose.yml   `docker compose up --build`
```
