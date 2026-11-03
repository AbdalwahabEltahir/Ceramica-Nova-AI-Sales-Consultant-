# Architecture

Open [architecture.html](architecture.html) in a browser for the interactive diagram. Use the
arrow keys or the step buttons to move through the stages.

## One chat turn (`POST /api/v1/chat`)

```
Customer message
  -> G1 input guard        regex + DeBERTa injection classifier
  -> Orchestrator          routes: spec question, more options, catalogue question, quote, discovery
  -> Discovery             extracts area, budget, indoor/outdoor, floor/wall; asks for what is missing
  -> Constraint engine     BR-01..BR-10, plain Python, no LLM
  -> Recommender           score, rank top 3, budget and stock aware
  -> Tools (Decimal)       calculate_tile_quantity, check_inventory, generate_quote (BR-07)
  -> Hybrid RAG            BM25 + dense vectors (LanceDB), reciprocal-rank fusion, cited
  -> LLM composer          Gemini tiers, rule-based fallback
  -> G2/G4 output guard    every SKU and $ must come from tools, else template
  -> Reply + visualizer + ledger
```

## Offline

- **Catalogue**: 15 SKUs with price, stock, slip rating and floor/wall/outdoor flags
  (`backend/app/models/catalog.py`).
- **RAG index**: documents pass an injection sanitiser, then BM25 and LanceDB embeddings
  (`rag/`).
- **Domain adaptation**: Qwen2.5-0.5B-Instruct with LoRA, evaluated against its own baseline
  (`training/`). Not wired into the live chat.
- **Quality**: 8 YAML scenario cases (`evaluation/`) and the pytest suite (`tests/`).

## Design rule

Rules decide, tools calculate, the language model only talks, and every answer is checked before
the customer sees it.
