import asyncio
import os
import re
from typing import TypedDict
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_neo4j import Neo4jGraph
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, END

load_dotenv()

# 1. Define the State dictionary that gets passed between nodes
class GraphState(TypedDict):
    question: str
    context: str
    generation: str
    compliance_status: str
    retry_count: int

# 1.5 Neo4j Connection
graph = Neo4jGraph(
    url=os.environ["NEO4J_URI"],
    username=os.environ["NEO4J_USERNAME"],
    password=os.environ["NEO4J_PASSWORD"],
    database=os.environ.get("NEO4J_DATABASE", "neo4j")
)

llm = ChatGoogleGenerativeAI(model="gemini-3.5-flash-lite", temperature=0, max_retries=0)

# Small, cheap model used only as a second-layer compliance judge (see llm_flags_money)
judge_llm = ChatGoogleGenerativeAI(model="gemini-3.5-flash-lite", temperature=0, max_retries=0)

MAX_RETRIES = 3
FALLBACK_ANSWER = "I can't answer that without disclosing restricted financial information"

# Each pattern flags one way of writing a money amount (matched case-insensitively).
MONEY_PATTERNS = [
    r"\$\s?\d",                                                    # $75,000  $75000  $1.5  and shorthand like $75k / $2M
    r"₹\s?\d",                                                      # rupee symbol: ₹50,000
    r"\b(?:rs\.?|inr|usd)\s?\d",                                    # code before number: Rs 500, Rs. 5,00,000, INR 5000, USD 75,000
    r"\d\s?(?:k|m|mm|bn|thousand|million|billion)?\s?(?:dollars|usd|rupees|inr)\b",  # number before word: 75,000 dollars, 75k USD
]

# --- NODES ---

def retrieve(state: GraphState):
    print("---NODE: RETRIEVING CONTEXT FROM NEO4J---")
    question = state["question"]

    # Match on meaningful keywords instead of the full sentence.
    tokens = [
        t for t in re.findall(r"[a-zA-Z0-9]+", question.lower())
        if len(t) > 2 and t not in {
            "who", "what", "when", "where", "why", "how", "does", "is", "are",
            "the", "and", "for", "with", "from", "that", "this", "about",
            "into", "your", "have", "has", "was", "were", "audited", "audit"
        }
    ]

    # Fallback to all words if aggressive filtering leaves nothing.
    if not tokens:
        tokens = re.findall(r"[a-zA-Z0-9]+", question.lower())

    query = """
    MATCH (n)-[r]->(m)
    WITH n, r, m,
         toLower(coalesce(n.id, '')) AS n_id,
         toLower(coalesce(m.id, '')) AS m_id,
         toLower(type(r)) AS rel,
         $tokens AS tokens
    WITH n, r, m, n_id, m_id, rel,
         reduce(score = 0, token IN tokens | 
           score +
           CASE
             WHEN n_id CONTAINS token OR m_id CONTAINS token OR rel CONTAINS token
             THEN 1 ELSE 0
           END
         ) AS match_score
    WHERE match_score > 0
    RETURN n.id AS source, type(r) AS relationship, m.id AS target, match_score
    ORDER BY match_score DESC
    LIMIT $limit
    """
    result = graph.query(query, params={"tokens": tokens, "limit": 20})

    # One-hop expansion: also pull edges touching the nodes of the matched facts, so
    # multi-hop questions (project -> role -> person) get the second hop.
    node_ids = list({r[k] for r in result for k in ("source", "target")})
    expand_query = """
    MATCH (n)-[r]->(m)
    WHERE n.id IN $ids OR m.id IN $ids
    RETURN n.id AS source, type(r) AS relationship, m.id AS target
    LIMIT $limit
    """
    expanded = graph.query(expand_query, params={"ids": node_ids, "limit": 40}) if node_ids else []

    # Matched facts first, then expansion facts; dedupe while keeping order.
    facts = list(dict.fromkeys(f"{r['source']} {r['relationship']} {r['target']}" for r in result + expanded))
    context = "\n".join(facts)
    print(f"Retrieved {len(result)} matched + {len(facts) - len(result)} expanded graph facts")
    return {"context": context}

async def generate(state: GraphState):
    print("---NODE: GENERATING ANSWER---")
    question = state["question"]
    context = state["context"]

    prompt = (
        "Answer the question based ONLY on this context. "
        "Never include financial figures or dollar amounts.\n"
        f"Context: {context}\nQuestion: {question}"
    )

    if state.get("compliance_status") == "FAIL":
        prompt += "\n\nMANAGER FEEDBACK: Your previous answer failed the compliance check because it included a dollar amount. You MUST NOT include any financial figures or dollar amounts in your revised answer."

    response = await llm.ainvoke(prompt)
    return {"generation": response.text}  # .text flattens list-style content blocks to a plain string

async def llm_flags_money(text: str) -> bool:
    """Second layer: ask a small LLM whether the text states a monetary amount in ANY form
    (words, shorthand, other currencies). Catches what the regex patterns miss."""
    response = await judge_llm.ainvoke(  # async so the event loop isn't blocked while waiting on the model
        "Does the following text state a specific monetary amount or financial figure in any form "
        "(digits, words like 'seventy-five thousand', shorthand, any currency)? "
        "Answer with exactly one word: YES or NO.\n\nText: " + text
    )
    return response.text.strip().upper().startswith("YES")

async def compliance_check(state: GraphState):
    print("---NODE: CHECKING COMPLIANCE---")
    generation = state["generation"]

    # Cheap deterministic regex first; the LLM judge only runs if the regex found nothing.
    has_money = any(re.search(p, generation, re.IGNORECASE) for p in MONEY_PATTERNS) or await llm_flags_money(generation)
    status = "FAIL" if has_money else "PASS"
    retry_count = state.get("retry_count", 0) + (1 if status == "FAIL" else 0)

    print(f"Compliance Status: {status}")
    update = {"compliance_status": status, "retry_count": retry_count}
    if status == "FAIL" and retry_count > MAX_RETRIES:
        # Retries exhausted: never surface the non-compliant text; status stays FAIL.
        update["generation"] = FALLBACK_ANSWER
    return update

# --- CONDITIONAL EDGE LOGIC ---

def route_compliance(state: GraphState):
    if state["compliance_status"] == "PASS":
        print("---ROUTE: COMPLIANT, ENDING---")
        return "end"
    if state.get("retry_count", 0) > MAX_RETRIES:
        print("---ROUTE: MAX RETRIES EXCEEDED, ENDING---")
        return "end"
    else:
        print("---ROUTE: FAILED COMPLIANCE, RE-GENERATING---")
        return "generate"

# --- BUILD THE GRAPH ---

workflow = StateGraph(GraphState)

# Add our three nodes
workflow.add_node("retrieve", retrieve)
workflow.add_node("generate", generate)
workflow.add_node("compliance", compliance_check)

# Define the flow
workflow.set_entry_point("retrieve")
workflow.add_edge("retrieve", "generate")
workflow.add_edge("generate", "compliance")

# The conditional routing
workflow.add_conditional_edges(
    "compliance",
    route_compliance,
    {
        "end": END,
        "generate": "generate" # Loop back if it fails
    }
)

# Compile the engine with in-memory checkpointing for thread state
app = workflow.compile(checkpointer=MemorySaver())

# --- EXECUTE ---
# if __name__ == "__main__":
#     inputs = {"question": "How much money does Project Phoenix involve and who manages it?"}
#     print("Starting LangGraph Agent...\n")

#     config = {"configurable": {"thread_id": "local-cli-thread"}}
#     final_state = asyncio.run(app.ainvoke(inputs, config=config))

#     print("\nFINAL OUTPUT:")
#     print(final_state["generation"])
