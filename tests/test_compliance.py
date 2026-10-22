"""Unit tests for compliance_check and route_compliance. No LLM or Neo4j calls."""
import os
import sys
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
for key in ("NEO4J_URI", "NEO4J_USERNAME", "NEO4J_PASSWORD"):
    os.environ.setdefault(key, "unused")

# agent.py builds a Neo4j connection and an LLM client at import time; stub both out.
with patch.dict(sys.modules, {"langchain_neo4j": MagicMock()}), patch("langchain_google_genai.ChatGoogleGenerativeAI"):
    import agent


def check(text, retry_count=0):
    return agent.compliance_check({"generation": text, "retry_count": retry_count})


def test_pass_case():
    result = check("Alice Chen must audit Project Phoenix.")
    assert result["compliance_status"] == "PASS"
    assert result["retry_count"] == 0
    assert "generation" not in result  # answer left untouched


def test_fail_case_increments_retry():
    result = check("Project Phoenix involves $75,000.", retry_count=1)
    assert result["compliance_status"] == "FAIL"
    assert result["retry_count"] == 2


@pytest.mark.parametrize("text", [
    "The budget is $75,000.",
    "The budget is $ 75,000.",
    "The budget is $75k.",
    "The budget is $1.5M.",
    "The budget is ₹50,000.",
    "The budget is Rs 50,000.",
    "The budget is Rs. 5,00,000.",
    "The budget is INR 50000.",
    "The budget is USD 75,000.",
    "The budget is 75,000 dollars.",
    "The budget is 75k USD.",
    "The budget is 2 million dollars.",
])
def test_money_formats_fail(text):
    assert check(text)["compliance_status"] == "FAIL"


@pytest.mark.parametrize("text", [
    "Bob Smith manages Project Phoenix.",
    "It was approved in 2024 by 3 teams.",
    "Work hours are 9 to 5.",
])
def test_non_money_passes(text):
    assert check(text)["compliance_status"] == "PASS"


def test_route_pass_ends():
    assert agent.route_compliance({"compliance_status": "PASS", "retry_count": 0}) == "end"


def test_route_fail_with_retries_left_regenerates():
    assert agent.route_compliance({"compliance_status": "FAIL", "retry_count": 1}) == "generate"
    assert agent.route_compliance({"compliance_status": "FAIL", "retry_count": agent.MAX_RETRIES}) == "generate"


def test_exhausted_retries_returns_fallback_and_ends():
    # Last allowed attempt (retry_count 3 -> 4) fails: checker swaps in the fallback.
    result = check("It costs $75,000.", retry_count=agent.MAX_RETRIES)
    assert result["compliance_status"] == "FAIL"  # stays visible
    assert result["generation"] == agent.FALLBACK_ANSWER
    assert "$" not in result["generation"]
    state = {"compliance_status": result["compliance_status"], "retry_count": result["retry_count"]}
    assert agent.route_compliance(state) == "end"


def test_fallback_not_applied_while_retries_remain():
    assert "generation" not in check("It costs $75,000.", retry_count=agent.MAX_RETRIES - 1)
