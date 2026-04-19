import os
import time
from typing import Any

import requests
import streamlit as st


BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000/ask")
REQUEST_TIMEOUT_SECONDS = 60


def call_backend(question: str) -> dict[str, Any]:
    response = requests.post(
        BACKEND_URL,
        json={"question": question},
        timeout=REQUEST_TIMEOUT_SECONDS,
    )
    response.raise_for_status()
    return response.json()


def render_metrics(latency: float | None, compliance_status: str | None, thread_id: str | None) -> None:
    compliance = (compliance_status or "UNKNOWN").upper()
    is_pass = compliance == "PASS"
    badge_color = "#16a34a" if is_pass else "#dc2626"
    badge_bg = "#dcfce7" if is_pass else "#fee2e2"

    st.markdown(
        f"""
        <div style="margin-top: 0.5rem; padding: 0.6rem 0.8rem; border: 1px solid #e5e7eb; border-radius: 0.5rem;">
            <div style="display: flex; gap: 1rem; flex-wrap: wrap; align-items: center;">
                <span><strong>Latency:</strong> {latency if latency is not None else "N/A"}s</span>
                <span>
                    <strong>Compliance:</strong>
                    <span style="margin-left: 0.25rem; color: {badge_color}; background: {badge_bg}; border-radius: 999px; padding: 0.15rem 0.5rem; font-weight: 600;">
                        {compliance}
                    </span>
                </span>
                <span><strong>Thread ID:</strong> {thread_id or "N/A"}</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


st.set_page_config(page_title="GraphRAG Interview Console", page_icon=":robot_face:", layout="wide")
st.title("GraphRAG Interview Console")
st.caption("FastAPI + LangGraph + Neo4j presentation layer")

with st.sidebar:
    st.header("System Diagnostics")
    st.markdown(
        """
        - Backend orchestration: **LangGraph multi-agent pipeline**
        - Retrieval layer: **Neo4j graph-backed RAG**
        - API endpoint: `POST /ask`
        - Frontend role: visualize reasoning flow + response metrics
        """
    )
    st.caption(f"Configured backend URL: {BACKEND_URL}")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant":
            render_metrics(msg.get("latency"), msg.get("compliance_status"), msg.get("thread_id"))

prompt = st.chat_input("Ask a question about your knowledge graph...")

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        try:
            with st.status("Running LangGraph pipeline...", expanded=True) as status:
                st.write("Retrieving from Neo4j...")
                time.sleep(0.35)
                st.write("Generating answer...")
                payload = call_backend(prompt)
                st.write("Checking compliance...")
                time.sleep(0.25)
                status.update(label="Pipeline complete", state="complete", expanded=False)

            answer = payload.get("answer", "No answer returned.")
            latency = payload.get("latency")
            compliance_status = payload.get("compliance_status", "UNKNOWN")
            thread_id = payload.get("thread_id")

            st.markdown(answer)
            render_metrics(latency, compliance_status, thread_id)

            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": answer,
                    "latency": latency,
                    "compliance_status": compliance_status,
                    "thread_id": thread_id,
                }
            )
        except requests.exceptions.RequestException:
            error_message = (
                "Backend is currently unreachable. Confirm FastAPI is running at "
                f"`{BACKEND_URL}` and try again."
            )
            st.error(error_message)
            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": error_message,
                    "latency": None,
                    "compliance_status": "FAIL",
                    "thread_id": None,
                }
            )
