import os
import time
from typing import Any

import requests
import streamlit as st


BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000/ask")
REQUEST_TIMEOUT_SECONDS = 60


def call_backend(question: str, thread_id: str | None = None) -> dict[str, Any]:
    payload = {"question": question}
    if thread_id:
        payload["thread_id"] = thread_id
        
    response = requests.post(
        BACKEND_URL,
        json=payload,
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
        <div style="margin-top: 0.5rem; padding: 0.75rem 1rem; border: 1px solid rgba(255,255,255,0.1); border-radius: 0.75rem; background: rgba(30,41,59,0.7); backdrop-filter: blur(10px); box-shadow: 0 4px 6px rgba(0,0,0,0.2);">
            <div style="display: flex; gap: 1.25rem; flex-wrap: wrap; align-items: center; color: #e2e8f0; font-size: 0.9rem;">
                <span><strong style="color: #94a3b8;">Latency:</strong> {latency if latency is not None else "N/A"}s</span>
                <span>
                    <strong style="color: #94a3b8;">Compliance:</strong>
                    <span style="margin-left: 0.25rem; color: {badge_color}; background: {badge_bg}; border-radius: 999px; padding: 0.15rem 0.6rem; font-weight: 600; font-size: 0.85rem; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
                        {compliance}
                    </span>
                </span>
                <span><strong style="color: #94a3b8;">Thread ID:</strong> {thread_id or "N/A"}</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


st.set_page_config(page_title="GraphRAG Interview Console", page_icon=":robot_face:", layout="wide")

# Inject Custom CSS for Premium Dark Aesthetics
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    /* Dark Premium Background */
    .stApp {
        background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 100%);
        color: #f8fafc;
    }
    
    /* Glassmorphism Sidebar */
    [data-testid="stSidebar"] {
        background: rgba(15, 23, 42, 0.6) !important;
        backdrop-filter: blur(12px) !important;
        border-right: 1px solid rgba(255, 255, 255, 0.05) !important;
    }
    
    /* Gradient Title */
    h1 {
        background: -webkit-linear-gradient(45deg, #38bdf8, #818cf8, #c084fc);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800 !important;
        letter-spacing: -1px;
    }
    
    /* Styled Chat Messages */
    .stChatMessage {
        background: rgba(30, 41, 59, 0.4) !important;
        border-radius: 12px !important;
        padding: 1.5rem !important;
        border: 1px solid rgba(255, 255, 255, 0.05) !important;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1) !important;
    }
    
    /* Floating Chat Input */
    .stChatInputContainer {
        border-radius: 20px !important;
        background: rgba(30, 41, 59, 0.9) !important;
        backdrop-filter: blur(10px) !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        box-shadow: 0 10px 25px rgba(0, 0, 0, 0.5) !important;
        padding-bottom: 2px !important;
    }
    
    /* Hide some default Streamlit decorations */
    header[data-testid="stHeader"] {
        background: transparent !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

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
if "thread_id" not in st.session_state:
    st.session_state.thread_id = None

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
                payload = call_backend(prompt, st.session_state.thread_id)
                st.write("Checking compliance...")
                time.sleep(0.25)
                status.update(label="Pipeline complete", state="complete", expanded=False)

            answer = payload.get("answer", "No answer returned.")
            latency = payload.get("latency")
            compliance_status = payload.get("compliance_status", "UNKNOWN")
            thread_id = payload.get("thread_id")
            
            if thread_id:
                st.session_state.thread_id = thread_id

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
