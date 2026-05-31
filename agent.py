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

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite", temperature=0)

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

    # Format the graph output into readable text context
    context = "\n".join([f"{r['source']} {r['relationship']} {r['target']}" for r in result])
    print(f"Retrieved {len(result)} graph facts")
    return {"context": context}

async def generate(state: GraphState):
    print("---NODE: GENERATING ANSWER---")
    question = state["question"]
    context = state["context"]

    prompt = f"Answer the question based ONLY on this context.\nContext: {context}\nQuestion: {question}"
    response = await llm.ainvoke(prompt)
    return {"generation": response.content}

def compliance_check(state: GraphState):
    print("---NODE: CHECKING COMPLIANCE---")
    generation = state["generation"]

    dollar_amount_pattern = r"\$\s?\d{1,3}(?:,\d{3})*(?:\.\d{2})?"
    has_dollar_amount = bool(re.search(dollar_amount_pattern, generation))
    status = "FAIL" if has_dollar_amount else "PASS"
    retry_count = state.get("retry_count", 0) + (1 if status == "FAIL" else 0)

    print(f"Compliance Status: {status}")
    return {"compliance_status": status, "retry_count": retry_count}

# --- CONDITIONAL EDGE LOGIC ---

def route_compliance(state: GraphState):
    if state["compliance_status"] == "PASS":
        print("---ROUTE: COMPLIANT, ENDING---")
        return "end"
    if state.get("retry_count", 0) > 3:
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
