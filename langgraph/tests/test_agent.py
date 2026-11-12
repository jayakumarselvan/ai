"""Tests use a scripted fake chat model, so no real LLM is needed."""

from __future__ import annotations

import pytest
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.types import Command

from ecommerce_agent import store
from ecommerce_agent.graph import build_graph


class ScriptedLLM(GenericFakeChatModel):
    def bind_tools(self, tools, **kwargs):
        return self


def tool_call(name: str, args: dict, call_id: str) -> AIMessage:
    return AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": call_id}])


def make(script: list[AIMessage]):
    return build_graph(llm=ScriptedLLM(messages=iter(script)))


def cfg(customer="C1001", thread="t1"):
    return {"configurable": {"thread_id": thread, "customer_id": customer}}


@pytest.fixture(autouse=True)
def _reset():
    store.reset()


def ask(graph, text, config):
    return graph.invoke({"messages": [HumanMessage(text)]}, config)


def test_order_details():
    g = make([tool_call("get_order_details", {"order_id": "ORD-1001"}, "1"), AIMessage("done")])
    r = ask(g, "details of ORD-1001", cfg())
    assert "Wireless Headphones" in r["messages"][-2].content
    assert r["messages"][-1].content == "done"


def test_cannot_read_other_customers_order():
    g = make([tool_call("get_order_details", {"order_id": "ORD-2001"}, "1"), AIMessage("done")])
    r = ask(g, "details of ORD-2001", cfg("C1001"))
    assert r["messages"][-2].content == "Order not found."


def test_small_refund_no_approval():
    g = make([
        tool_call("return_product", {"order_id": "ORD-1001", "sku": "HP-100", "quantity": 1, "reason": "faulty"}, "1"),
        tool_call("request_refund", {"return_id": "RET-0001"}, "2"),
        AIMessage("refunded"),
    ])
    r = ask(g, "return headphones and refund", cfg())
    assert "__interrupt__" not in r
    assert store.RETURNS["RET-0001"]["status"] == "refunded"


def _big_refund_script():
    return [
        tool_call("return_product", {"order_id": "ORD-1002", "sku": "KB-MECH", "quantity": 1, "reason": "wrong"}, "1"),
        tool_call("request_refund", {"return_id": "RET-0001"}, "2"),
        AIMessage("final"),
    ]


def test_large_refund_requires_approval_then_approved():
    g = make(_big_refund_script())
    r = ask(g, "return keyboard and refund", cfg())
    assert r["__interrupt__"][0].value["amount"] == 149.0
    assert store.RETURNS["RET-0001"]["status"] == "requested"  # nothing refunded yet
    r = g.invoke(Command(resume={"approved": True}), cfg())
    assert store.RETURNS["RET-0001"]["status"] == "refunded"
    assert r["messages"][-1].content == "final"


def test_large_refund_declined():
    g = make(_big_refund_script())
    ask(g, "return keyboard and refund", cfg())
    r = g.invoke(Command(resume={"approved": False, "note": "policy"}), cfg())
    assert store.RETURNS["RET-0001"]["status"] == "requested"
    assert "DECLINED" in r["messages"][-2].content


def test_return_window_expired():
    out = store.create_return("C1001", "ORD-1005", "WC-HD", 1, "meh")
    assert "expired" in out


def test_cancel_only_when_processing():
    assert "cancelled" in store.cancel_order("C1001", "ORD-1004")
    assert "can no longer" in store.cancel_order("C1001", "ORD-1003")
