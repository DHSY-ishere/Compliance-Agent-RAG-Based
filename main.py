import time
from uuid import uuid4
import mlflow
from fastapi import FastAPI
from typing import Optional
from pydantic import BaseModel
from agent import app as graph_agent # Import our compiled LangGraph

# Initialize FastAPI
api = FastAPI(title="GraphRAG Compliance Agent")

# Define request body structure
class QueryRequest(BaseModel):
    question: str
    thread_id: Optional[str] = None

@api.post("/ask")
async def ask_agent(request: QueryRequest):
    start_time = time.time()

    # Starting MLFlow tracking
    with mlflow.start_run():
        mlflow.log_param("question", request.question)

        # Executing the LangGraph loop asynchronously with checkpoint thread id
        thread_id = request.thread_id or str(uuid4())
        inputs = {"question": request.question}
        config = {"configurable": {"thread_id": thread_id}}
        final_state = await graph_agent.ainvoke(inputs, config=config)

        # Calculating latency
        latency = time.time() - start_time
        mlflow.log_metric("latency_seconds", latency)
        mlflow.log_param("compliance_status", final_state["compliance_status"])
        mlflow.log_param("thread_id", thread_id)

        return {
            "answer": final_state["generation"],
            "latency": round(latency, 2),
            "compliance_status": final_state["compliance_status"],
            "thread_id": thread_id
        }
