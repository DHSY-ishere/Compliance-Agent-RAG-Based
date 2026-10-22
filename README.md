# Compliance-Agent-RAG-Based

A GraphRAG question-answering service with a compliance guardrail. It is a single **LangGraph workflow**: retrieve facts from a Neo4j knowledge graph, generate an answer with Gemini, run a **deterministic compliance check**, and regenerate (bounded) if the check fails. This is the *evaluator-optimizer* pattern, not a multi-agent orchestrator.

The guardrail enforces one business rule: **answers must not contain financial figures** (dollar/rupee amounts). It is a rule checker, **not a hallucination detector**; it does not verify that an answer is grounded in the retrieved context.

## Architecture

```
                 +-----------+      +-----------+      +--------------------+
 question  --->  | retrieve  | ---> | generate  | ---> | compliance_check   |
                 | (Neo4j)   |      | (Gemini)  |      | (regex, no LLM)    |
                 +-----------+      +-----------+      +--------------------+
                                          ^                     |
                                          |  FAIL and           | PASS, or retries exhausted
                                          |  retry_count <= 3   v
                                          +-------------- route_compliance ---> END
```

- `retrieve`: pulls graph facts relevant to the question (see below).
- `generate`: answers using only the retrieved context. The prompt already says "Never include financial figures or dollar amounts". After a FAIL it also receives explicit feedback.
- `compliance_check`: matches the answer against `MONEY_PATTERNS` ($, ₹, Rs/INR/USD, "75k", "75,000 dollars", ...). FAIL increments `retry_count`.
- `route_compliance`: loops back to `generate` on FAIL while `retry_count <= 3`; otherwise ends.
- **Retry exhaustion**: if the 4th attempt still fails, the answer is replaced by a safe fallback ("I can't answer that without disclosing restricted financial information") and `compliance_status` stays `FAIL` so the failure is visible in the API response and MLflow.
- Per-turn state (`retry_count`, `compliance_status`, `generation`, `context`) is reset on every request, because the `MemorySaver` checkpointer would otherwise carry it over within a `thread_id`.

Other pieces: `populate_graph.py` builds the graph from `data/*.txt` with `LLMGraphTransformer`; `main.py` serves FastAPI `/ask` and logs to MLflow; `app.py` is a Streamlit UI.

## How retrieval works

Not vector search. `retrieve` lowercases the question, strips stopwords, and keeps the remaining keywords. A single Cypher query then matches every edge `(n)-[r]->(m)` and scores it by how many keywords appear in the source id, target id, or relationship type. The top 20 edges (score > 0, highest first) are formatted as `source RELATIONSHIP target` lines and passed to the LLM as context.

## Evaluation

`golden_set.json` has 8 questions with expected answer substrings (including multi-hop "Who needs to audit Project Phoenix?", two questions whose natural answer is a dollar amount, and one with no answer in the data). `python eval.py` runs each through the graph and reports correctness, whether the *first* generation violated the rule (before the checker), final compliance pass rate, average retries, and average latency.

Results: **not yet run against the live stack.** Paste the output of `python eval.py` here:

| Metric | Result |
|---|---|
| Answer correctness | _TBD_ / 8 |
| First generation violated rule | _TBD_ / 8 |
| Final compliance pass rate | _TBD_ / 8 |
| Average retries | _TBD_ |
| Average latency | _TBD_ s |

Notes on the metric: correctness is a case-insensitive substring match against a list of acceptable strings, and for the dollar-amount questions a refusal or naming the manager counts as correct.

## Running

```
pip install -r requirements.txt
# .env: NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD, GOOGLE_API_KEY
python populate_graph.py      # build the graph from ./data
uvicorn main:api --reload     # API: POST /ask {"question": "...", "thread_id": "optional"}
streamlit run app.py          # UI
python -m pytest tests -q     # unit tests (no LLM or Neo4j needed)
python eval.py                # evaluation (needs live Neo4j + Gemini)
```

## Known limitations

- **Keyword retrieval scans all edges.** The Cypher query touches every relationship on each request, so it does not scale, and it misses paraphrases and synonyms (no embeddings).
- **Regex can miss paraphrased amounts**, e.g. "seventy-five thousand dollars" or "three quarters of a hundred thousand". It only catches the formats in `MONEY_PATTERNS`.
- **No groundedness check.** Nothing verifies that an answer is supported by the retrieved facts; a compliant answer can still be wrong.
- The retry limit (3) and fallback text are fixed constants; state is in-memory only (`MemorySaver`), so threads are lost on restart.
