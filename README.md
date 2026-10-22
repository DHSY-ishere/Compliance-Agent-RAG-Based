# Compliance-Agent-RAG-Based

A GraphRAG question-answering service with a compliance guardrail. It is a single **LangGraph workflow**: retrieve facts from a Neo4j knowledge graph, generate an answer with Gemini, run a **two-layer compliance check** (regex first, then a small LLM judge), and regenerate (bounded) if the check fails. This is the *evaluator-optimizer* pattern, not a multi-agent orchestrator.

The guardrail enforces one business rule: **answers must not contain financial figures** (dollar/rupee amounts). It is a rule checker, **not a hallucination detector**; it does not verify that an answer is grounded in the retrieved context.

## Architecture

```
                 +-----------+      +-----------+      +--------------------+
 question  --->  | retrieve  | ---> | generate  | ---> | compliance_check   |
                 | (Neo4j)   |      | (Gemini)  |      | (regex + LLM judge)|
                 +-----------+      +-----------+      +--------------------+
                                          ^                     |
                                          |  FAIL and           | PASS, or retries exhausted
                                          |  retry_count <= 3   v
                                          +-------------- route_compliance ---> END
```

- `retrieve`: pulls graph facts relevant to the question (see below).
- `generate`: answers using only the retrieved context. The prompt already says "Never include financial figures or dollar amounts". After a FAIL it also receives explicit feedback.
- `compliance_check`: first matches the answer against `MONEY_PATTERNS` ($, ₹, Rs/INR/USD, "75k", "75,000 dollars", ...). If the regex finds nothing, a small LLM judge (`llm_flags_money`) is asked whether the text states a monetary amount in any form (words, other formats). Either layer can FAIL; FAIL increments `retry_count`. The regex is the cheap deterministic floor; the judge is probabilistic, so it is a backstop, not a proof.
- `route_compliance`: loops back to `generate` on FAIL while `retry_count <= 3`; otherwise ends.
- **Retry exhaustion**: if the 4th attempt still fails, the answer is replaced by a safe fallback ("I can't answer that without disclosing restricted financial information") and `compliance_status` stays `FAIL` so the failure is visible in the API response and MLflow.
- Per-turn state (`retry_count`, `compliance_status`, `generation`, `context`) is reset on every request, because the `MemorySaver` checkpointer would otherwise carry it over within a `thread_id`.

Other pieces: `populate_graph.py` builds the graph from `data/*.txt` with `LLMGraphTransformer`; `main.py` serves FastAPI `/ask` and logs to MLflow; `app.py` is a Streamlit UI.

## How retrieval works

Not vector search. `retrieve` lowercases the question, strips stopwords, and keeps the remaining keywords. A single Cypher query then matches every edge `(n)-[r]->(m)` and scores it by how many keywords appear in the source id, target id, or relationship type. The top 20 edges (score > 0, highest first) are then **expanded by one hop**: a second query adds up to 40 edges touching any node of those matches, so multi-hop questions (project -> role -> person) get their second hop. Matched facts come first, then expansion facts, deduplicated, formatted as `source RELATIONSHIP target` lines and passed to the LLM as context.

## Evaluation

`golden_set.json` has 8 questions with expected answer substrings (including multi-hop "Who needs to audit Project Phoenix?", two questions whose natural answer is a dollar amount, and one with no answer in the data). `python eval.py` runs each through the graph and reports correctness, whether the *first* generation violated the rule (before the checker), final compliance pass rate, average retries, and average latency.

Results from `python eval.py --delay 15` (full output in `eval_output.txt`; `gemini-3.5-flash-lite` for generation and judging, 8 questions, one run):

| Metric | Result |
|---|---|
| Answer correctness | 7 / 8 |
| First generation violated rule | 0 / 8 |
| Final compliance pass rate | 8 / 8 |
| Average retries | 0.00 |
| Average latency | 11.65 s (includes free-tier rate-limit waits) |

How to read this honestly:

- **Results vary run to run** even at temperature 0. In an earlier run with the same code, one question (Project Titan budget) failed its first generation, was caught by the LLM judge, and passed after 1 retry (average retries 0.12); in this run it passed first time. Treat the numbers as indicative of an 8-question sample, not a benchmark.
- **The guardrail is rarely triggered.** Dollar amounts are now in the graph (`populate_graph.py` instructs the extractor to keep them), but the first-turn prompt already says "Never include financial figures", and the model mostly complies. The regenerate loop and the retry-exhaustion fallback are therefore exercised mainly by the unit tests, not by this eval.
- **The one wrong answer** is "Who needs to audit Project Phoenix?": the model answers "the Head Of Risk Management" even though the edge to Alice Chen is in the retrieved context. A single-hop answer where the multi-hop one was needed is a model reasoning failure, not a retrieval one. The same question was answered correctly in an earlier run on the larger `gemini-3.5-flash` (before its free-tier daily quota ran out), so the smaller model is the likely cause.
- **Retrieval changes took the eval from 5/8 to 7/8** (an earlier run, before one-hop expansion and amounts in the graph, had the CFO question retrieving 0 facts; it now passes). That comparison is across different models and data, so it is indicative only.
- Correctness is a case-insensitive substring match; for the dollar questions a refusal or naming the manager counts as correct, and the no-answer question accepts any "don't know" style phrase (I added "no mention" after seeing a correct answer worded that way).
- "First generation violated rule" is the checker's verdict (regex or judge) on the first generation.

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
- **Regex can miss paraphrased amounts** (e.g. "seventy-five thousand dollars"). The LLM judge covers many of these but is probabilistic and adds one extra model call per answer.
- **Retrieval depth is one hop.** Questions needing three or more hops will still miss facts.
- **No synonym matching.** "CFO" does not match "Chief Financial Officer".
- **Ingestion depends on an extra instruction** to keep dollar amounts; without it the extractor drops them.
- **No groundedness check.** Nothing verifies that an answer is supported by the retrieved facts; a compliant answer can still be wrong.
- The retry limit (3) and fallback text are fixed constants; state is in-memory only (`MemorySaver`), so threads are lost on restart.
