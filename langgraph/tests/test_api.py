"""API tests with a scripted fake LLM (no network / API key)."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage

from ecommerce_agent import store
from ecommerce_agent.api import create_app
from ecommerce_agent.graph import build_graph
from tests.test_agent import ScriptedLLM, tool_call

CUST_A = {"Authorization": "Bearer tok-a"}
CUST_B = {"Authorization": "Bearer tok-b"}
SUPPORT = {"Authorization": "Bearer support-secret"}


def make_client(script):
    graph = build_graph(llm=ScriptedLLM(messages=iter(script)))
    app = create_app(graph, {"tok-a": "C1001", "tok-b": "C1002"}, "support-secret")
    return TestClient(app)


@pytest.fixture(autouse=True)
def _reset():
    store.reset()


def big_refund_script():
    return [
        tool_call("return_product", {"order_id": "ORD-1002", "sku": "KB-MECH", "quantity": 1, "reason": "wrong"}, "1"),
        tool_call("request_refund", {"return_id": "RET-0001"}, "2"),
        AIMessage("Your refund was processed."),
    ]


def test_requires_auth():
    c = make_client([])
    assert c.post("/chat", json={"message": "hi"}).status_code == 401
    assert c.post("/chat", json={"message": "hi"}, headers={"Authorization": "Bearer nope"}).status_code == 401
    assert c.get("/support/approvals", headers=CUST_A).status_code == 401  # customer token is not a support key


def test_large_refund_waits_for_support_then_approved():
    c = make_client(big_refund_script())
    r = c.post("/chat", json={"message": "return keyboard and refund"}, headers=CUST_A)
    assert r.status_code == 200 and r.json()["status"] == "pending_approval"
    tid = r.json()["thread_id"]
    assert store.RETURNS["RET-0001"]["status"] == "requested"

    items = c.get("/support/approvals", headers=SUPPORT).json()
    assert len(items) == 1 and items[0]["amount"] == 149.0 and items[0]["thread_id"] == tid

    # customer cannot keep chatting while waiting, and others cannot see the thread
    assert c.post("/chat", json={"message": "hello", "thread_id": tid}, headers=CUST_A).status_code == 409
    assert c.get(f"/chat/{tid}", headers=CUST_B).status_code == 404

    d = c.post(f"/support/approvals/{tid}/decision", json={"approved": True}, headers=SUPPORT)
    assert d.status_code == 200 and d.json()["status"] == "completed"
    assert store.RETURNS["RET-0001"]["status"] == "refunded"
    assert c.get("/support/approvals", headers=SUPPORT).json() == []
    assert c.get(f"/chat/{tid}", headers=CUST_A).json()["reply"] == "Your refund was processed."


def test_large_refund_declined():
    c = make_client(big_refund_script())
    tid = c.post("/chat", json={"message": "refund keyboard"}, headers=CUST_A).json()["thread_id"]
    d = c.post(f"/support/approvals/{tid}/decision", json={"approved": False, "note": "policy"}, headers=SUPPORT)
    assert d.status_code == 200
    assert store.RETURNS["RET-0001"]["status"] == "requested"


def test_decision_without_pending_is_404_and_customer_cannot_decide():
    c = make_client(big_refund_script())
    tid = c.post("/chat", json={"message": "refund keyboard"}, headers=CUST_A).json()["thread_id"]
    assert c.post(f"/support/approvals/{tid}/decision", json={"approved": True}, headers=CUST_A).status_code == 401
    assert c.post("/support/approvals/unknown/decision", json={"approved": True}, headers=SUPPORT).status_code == 404


def test_small_refund_completes_immediately():
    c = make_client([
        tool_call("return_product", {"order_id": "ORD-1001", "sku": "HP-100", "quantity": 1, "reason": "faulty"}, "1"),
        tool_call("request_refund", {"return_id": "RET-0001"}, "2"),
        AIMessage("Refunded."),
    ])
    r = c.post("/chat", json={"message": "refund headphones"}, headers=CUST_A)
    assert r.json()["status"] == "completed"
    assert store.RETURNS["RET-0001"]["status"] == "refunded"


def test_test_ui_is_served():
    r = make_client([]).get("/")
    assert r.status_code == 200 and "text/html" in r.headers["content-type"]
