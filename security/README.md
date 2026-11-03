# Security

Threats considered, where each is handled in code, and where it is tested.

| Threat | Control | Code | Tested by |
|---|---|---|---|
| Prompt injection (role override, prompt extraction, false authority such as "management approved", cross-customer requests), English and Arabic | G1 input guard: regex patterns plus a DeBERTa classifier. A flagged message never reaches the LLM; the real catalogue price is still returned | `backend/app/services/guardrails/input_guard.py` | `tests/test_app.py` (GuardrailTests, scenario 7), `evaluation/cases/E-SEC-001/002.yaml` |
| Indirect injection through retrieved documents | Ingest sanitiser strips instruction-like text before indexing | `rag/ingest.py` | `tests/test_rag_and_llm.py` |
| Model invents a price, SKU or availability | G4 output guard checks every SKU and dollar amount against tool results, blocks scarcity language and delivery promises, replaces the reply with a template on failure | `backend/app/services/guardrails/output_guard.py` | `tests/test_app.py` (GuardrailTests, scenario 8) |
| Model tries to call tools it should not, or alter prices | G3 tool whitelist; prices and quote owner always come from the server, never from the request | `backend/app/services/guardrails/` , `services/tools/` | `tests/test_app.py` |
| Unsafe recommendation (wrong slip rating, wall tile on a floor) | Hard constraints BR-01 to BR-10 in plain Python, independent of any prompt | `backend/app/models/rules.py` | `tests/test_app.py` (RuleTests, scenarios 1, 5) |
| Unauthorised discount | BR-07: a discount above 12% becomes PENDING_APPROVAL and is never described as approved | `backend/app/models/rules.py`, `services/tools/quotation.py` | `tests/test_app.py` (RuleTests, scenario 5 in the demo) |
| Cross-session data access | Sessions and quotes are scoped to the session cookie | `backend/app/models/` | `tests/test_app.py` |
| Abuse and cost | Per-client rate limit of 30 requests per minute | `backend/app/` | `/metrics` counters |
| Cross-site scripting in chat | UI renders with `textContent` only, never `innerHTML` | `frontend/static/js/chat.js` | manual review |
| Secrets in the repo | `.env` is git-ignored; only `.env.example` is committed | `.gitignore` | n/a |

Attack samples used for testing are in `data/injection_fixtures.json`.

## Known gaps

- `POST /api/v1/quotes/<id>/approve` has no staff authentication yet. Anyone who can reach the
  API can approve a pending quote. A real deployment needs a staff role check.
- Sessions and quotes are in memory and are lost on restart.
- The DeBERTa classifier needs a Hugging Face key. Without it the guard degrades to regex only.
- No formal red-team suite beyond the fixtures and the two injection evaluation cases.
