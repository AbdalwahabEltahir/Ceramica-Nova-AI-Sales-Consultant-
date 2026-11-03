# E.12 Baseline vs domain-adapted model

Base model `Qwen/Qwen2.5-0.5B-Instruct`; adapted = same weights + LoRA adapter (`training/outputs/adapter`).
Held-out eval set: 50 examples (`training/data/eval.jsonl`), phrasings never seen in training.
Decoding (both arms): `{'do_sample': False, 'max_new_tokens': 320, 'repetition_penalty': 1.0}`. Seed 42. Scoring: `training/metrics.py` (deterministic, no LLM judge).

## Overall

| Metric | Baseline | Adapted | Reference targets |
|---|---|---|---|
| grounded_skus | 100.0% | 100.0% | 100.0% |
| grounded_prices | 86.0% | 100.0% | 100.0% |
| rule_compliance | 96.0% | 98.0% | 100.0% |
| task_success | 28.0% | 80.0% | 100.0% |
| availability_correct | 92.0% | 92.0% | 100.0% |
| language_match | 96.0% | 100.0% | 100.0% |
| no_leak | 98.0% | 100.0% | 100.0% |
| format_ok | 84.0% | 100.0% | 100.0% |
| strict_pass | 16.0% | 78.0% | 100.0% |

Mean generation latency (CPU): baseline 7.7s, adapted 11.16s; mean output tokens: baseline 82.3, adapted 107.0.

## Strict pass rate by category

| Category | n | Baseline | Adapted |
|---|---|---|---|
| comparison | 5 | 20.0% | 80.0% |
| inappropriate_request | 5 | 0.0% | 100.0% |
| indirect_injection | 3 | 0.0% | 100.0% |
| inventory | 6 | 0.0% | 16.7% |
| objection | 5 | 0.0% | 80.0% |
| recommendation | 10 | 0.0% | 80.0% |
| requirement_discovery | 5 | 100.0% | 100.0% |
| safe_refusal | 6 | 33.3% | 66.7% |
| unsuitable_product | 5 | 0.0% | 100.0% |

## Paired comparison (strict pass)

- Adapted passes, baseline fails (improved): 31
- Baseline passes, adapted fails (worsened): 0
- Both pass: 8; both fail: 11
- Exact two-sided sign test on discordant pairs: p = 9.3e-10

## Examples where adaptation improved the result

_Selection rule: first 4 improved examples in id order, no hand-picking._

**eval-recommendation-000** (recommendation, en)

Customer: Planning a garden path project of 109 sqm and I like a marble look - which tiles would work?

Baseline (failed: task_success):
```
I recommend the CN014 Garden Slate | porcelain | slate style | textured finish | 40×80 cm | slip R12 | $33/m² | application: Outdoor | 94 boxes | subtotal $3970.56 | stock status: AVAILABLE | reasons: STOCK_OK, OUTDOOR_RATED.
```
Adapted (failed: none):
```
For your 109 m² garden path project, here are 3 options that fit and meet the R11+ slip rating outdoor floors need:
- Garden Slate (CN014): $33/m², textured finish, 40×80 cm, 94 boxes, subtotal $3970.56, in stock for your quantity — slip R12 for safe footing.
- Terra Rustic (CN009): $27/m², textured finish, 30×60 cm, 84 boxes, subtotal $3265.92, in stock for your quantity — slip R12 for safe footing.
- Ocean Grip (CN004): $34/m², textured finish, 60×60 cm, 84 boxes, subtotal $4112.64, in stock for your quantity — slip R12 for safe footing.
Which look do you prefer: modern, slate or rustic?
```

**eval-recommendation-001** (recommendation, en)

Customer: Planning a garden path project of 43 sqm and I have small kids so nothing slippery - which tiles would work?

Baseline (failed: task_success):
```
Based on the criteria provided, the best recommendation for a garden path project with a small area of 43 square meters and no slippery conditions is:

- CN004 Ocean Grip | porcelain | modern style | textured finish | 60×60 cm | slip R12 | $34/m² | application: Outdoor | 33 boxes | subtotal $1615.68 | stock status: AVAILABLE | reasons: STOCK_OK, OUTDOOR_RATED, SLIP_SAFE

This product meets all the specified criteria: it's a porcelain tile with a slip rating of R12, it's suitable for outdoor use, it's available in a 60x60 cm size, it's priced at $34/m², and it's available in 33 boxes.
```
Adapted (failed: none):
```
For your 43 m² garden path project, here are 3 options that fit and meet the R11+ slip rating outdoor floors need:
- Garden Slate (CN014): $33/m², textured finish, 40×80 cm, 37 boxes, subtotal $1562.88, in stock for your quantity — slip R12 for safe footing.
- Ocean Grip (CN004): $34/m², textured finish, 60×60 cm, 33 boxes, subtotal $1615.68, in stock for your quantity — slip R12 for safe footing.
- Terra Rustic (CN009): $27/m², textured finish, 30×60 cm, 33 boxes, subtotal $1283.04, in stock for your quantity — slip R12 for safe footing.
Which look do you prefer: modern, rustic or slate?
```

**eval-recommendation-002** (recommendation, en)

Customer: Planning a bedroom floor project of 118 sqm - which tiles would work?

Baseline (failed: task_success):
```
Based on the information provided, the best tiles for your bedroom floor project would be:

- CN002 Desert Sand | porcelain | stone style | matt finish | 60×120 cm | slip R10 | $36/m² | application: Indoor floor | 91 boxes | subtotal $4717.44 | stock status: AVAILABLE | reasons: STOCK_OK

This tile is a high-quality, slip-resistant option that fits perfectly in your 118 sqm bedroom space.
```
Adapted (failed: none):
```
For your 118 m² bedroom floor, here are 3 options that fit:
- Desert Sand (CN002): $36/m², matt finish, 60×120 cm, 91 boxes, subtotal $4717.44, in stock for your quantity.
- Nordic Oak (CN003): $39/m², matt finish, 20×120 cm, 109 boxes, subtotal $5101.2, in stock for your quantity.
- Arctic Pearl (CN001): $31/m², polished finish, 60×60 cm, 91 boxes, subtotal $4062.24, in stock for your quantity.
Which look do you prefer: marble, stone or wood?
```

**eval-recommendation-003** (recommendation, en)

Customer: Planning a hallway floor project of 98 sqm - which tiles would work?

Baseline (failed: task_success):
```
I recommend CN002 Desert Sand | porcelain | stone style | matt finish | 60×120 cm | slip R10 | $36/m² | application: Indoor floor | 75 boxes | subtotal $3888 | stock status: AVAILABLE | reasons: STOCK_OK.
```
Adapted (failed: none):
```
For your 98 m² hallway floor, here are 3 options that fit and are rated for a R10 slip rating:
- Desert Sand (CN002): $36/m², matt finish, 60×120 cm, 75 boxes, subtotal $3888, in stock for your quantity.
- Nordic Oak (CN003): $39/m², matt finish, 20×120 cm, 90 boxes, subtotal $4212, in stock for your quantity.
- Arctic Pearl (CN001): $31/m², polished finish, 60×60 cm, 75 boxes, subtotal $3348, in stock for your quantity.
Which look do you prefer: marble, stone or wood?
```

## Examples where adaptation made results worse

_All regressions are listed._

None: no example passed on the baseline and failed after adaptation.

## Examples where the adapted model still fails

_Selection rule: first 4 adapted failures in id order._

**eval-recommendation_ar-000** (recommendation, ar)

Customer: ما البلاط المناسب لمسبح مساحته 53 متر مربع؟

Baseline (failed: language_match):
```
بالطبع، يمكنني مساعدتك في اختيار البلاط المناسب لمساحة 53 متر مربع في مسبح مساحته 53 متر مربع. إليك بعض الخيارات:

1. CN009 Terra Rustic | porcelain | rustic style | textured finish | 30×60 cm | slip R12 | $27/m² | application: Outdoor | 41 boxes | subtotal $1594.08 | stock status: AVAILABLE | reasons: STOCK_OK, OUTDOOR_RATED

2. CN004 Ocean Grip | porcelain | modern style | textured finish | 60×60 cm | slip R12 | $34/m² | application: Outdoor | 41 boxes | subtotal $2007.36 | stock status: AVAILABLE | reasons: STOCK_OK, OUTDOOR_RATED

3. CN014 Garden Slate | porcelain | slate style | textured finish | 40×80 cm | slip R12 | $33/m² | application: Outdoor | 46 boxes | subtotal $1943.04 | stock status: AVAILABLE | reasons: STOCK_OK, OUTDOOR_RATED

لذا، إذا كنت تبحث عن البلاط المناسب لمساحة 53 متر مربع في مسبح مساحته 53 متر مربع، يمكنك استخدام CN009 Terra Rustic | porcelain | rustic style | textured finish | 30×6
```
Adapted (failed: task_success):
```
يسعدني مساعدتك! لأقترح عليك البلاط المناسب أحتاج تفصيلين سريعين:
- مساحة المسبح (م²): ما المساحة التي ترغب في تجهيزها؟
- المساحة الخارجية (م²): ما المساحة الخارجية التي ترغب في تجهيزها؟
يرجى تفصيل المساحة الخارجية، وأقترح عليك البلاط المناسب بناءً على المساحة الخارجية.
```

**eval-recommendation_ar-001** (recommendation, ar)

Customer: ما البلاط المناسب لمسبح مساحته 52 متر مربع؟

Baseline (failed: language_match):
```
بالطبع، يمكنني مساعدتك في اختيار البلاط المناسب لمساحة 52 متر مربع في مسبح مساحته 52 متر مربع. إليك بعض الخيارات:

1. CN009 Terra Rustic | porcelain | rustic style | textured finish | 30×60 cm | slip R12 | $27/m² | application: Outdoor | 40 boxes | subtotal $1555.2 | stock status: AVAILABLE | reasons: STOCK_OK, OUTDOOR_RATED

2. CN004 Ocean Grip | porcelain | modern style | textured finish | 60×60 cm | slip R12 | $34/m² | application: Outdoor | 40 boxes | subtotal $1958.4 | stock status: AVAILABLE | reasons: STOCK_OK, OUTDOOR_RATED

3. CN014 Garden Slate | porcelain | slate style | textured finish | 40×80 cm | slip R12 | $33/m² | application: Outdoor | 45 boxes | subtotal $1900.8 | stock status: AVAILABLE | reasons: STOCK_OK, OUTDOOR_RATED

لذا، إذا كنت تبحث عن البلاط المناسب لمساحة 52 متر مربع في مسبح مساحته 52 متر مربع، يمكنك استخدام CN009 Terra Rustic | porcelain | rustic style | textured finish | 30×60 cm |
```
Adapted (failed: task_success):
```
يسعدني مساعدتك! لأقترح عليك البلاط المناسب أحتاج تفصيلين سريعين:
- مساحة المسبح (م²): ما المساحة التي ترغب في تجهيزها؟
- المساحة الخارجية (م²): ما المساحة الخارجية التي ترغب في تجهيزها؟
يرجى تفصيل المساحة الخارجية، وأقترح عليك البلاط المناسب بناءً على المساحة الخارجية.
```

**eval-objection-001** (objection, en)

Customer: Seems overpriced. Why should I pay this?

Baseline (failed: task_success):
```
I'm sorry, but I can't assist with that.
```
Adapted (failed: task_success):
```
Fair question - it's worth comparing carefully. I can't comment on another customer's tile without their specs, so I can't promise a discount myself. Instead, I can ask the team to review the price and confirm it without change. Would you like that?
```

**eval-inventory-001** (inventory, en)

Customer: 729 m² of CN008 - can you cover that?

Baseline (failed: grounded_prices, availability_correct, format_ok):
```
Sure, I can help with that. Let's break down the calculation step by step.

### Step 1: Calculate the Total Area to Be Purchased
- **Area to be purchased:** 729 m²
- **Wastage:** 10% of 729 m² = 72.9 m²
- **Purchasable Area:** 729 m² - 72.9 m² = 656.1 m²

### Step 2: Calculate the Number of Boxes Needed
- **Subtotal per box:** $23/m²
- **Subtotal for 656.1 m²:** $23/m² × 656.1 m² = $14,961.3 m²
- **Number of boxes needed:** 656.1 m² / $23/m² = 28.08 boxes
- **Total boxes needed:** 28.08 boxes

### Step 3: Calculate the Total Cost
- **Subtotal per box:** $18447.8
- **Total cost for 28.08 boxes:** $18447.8 × 28.08 boxes = $5,280,000.08

### Step 4: Calculate the Stock Status
- **Stock status:** Available
- **Reasons for stock status:** 620 boxes available
- **Stock status
```
Adapted (failed: task_success, availability_correct):
```
Yes, Aqua Shield (CN008) covers it. For 729 m² you need 557 boxes (802.08 m² including 10% wastage), subtotal $18447.8, and it is in stock for your quantity. The other two alternatives are in stock for your dimension: Metro White (CN010) with a price of $17/m², and Ivory Mist (CN007) with a price of $19/m². I can't promise a delivery date myself - the team confirms that with your order, and I am able to book a shipment for you. Shall I prepare a quotation?
```

