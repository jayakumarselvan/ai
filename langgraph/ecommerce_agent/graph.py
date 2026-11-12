"""LangGraph definition: agent <-> tools loop with a human-approval gate for large refunds."""

from __future__ import annotations

from typing import Annotated, Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, SystemMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.types import interrupt
from typing_extensions import TypedDict

from . import store
from .config import get_settings
from .tools import ALL_TOOLS, REFUND_TOOL_NAME

SYSTEM_PROMPT = """You are a friendly, concise customer-support assistant for an online store.
You help the signed-in customer with: order details and status, shipment tracking, order
cancellation, product returns, and refunds.

Rules:
- Use the tools for every fact about orders/returns. Never invent order data.
- If the customer has not given an order id, call list_my_orders or ask for it.
- To refund: first create the return with return_product, then call request_refund with the
  returned RET id. Never state a refund amount yourself - use the tool result.
- Refunds above the approval threshold are automatically sent to a human support agent. If a
  tool says the refund was approved or declined, relay that. Do not promise approval beforehand.
- Before returning an item or cancelling an order, make sure the customer clearly asked for it.
- Only discuss the signed-in customer's own orders. Decline unrelated requests politely."""


class State(TypedDict):
    messages: Annotated[list, add_messages]


def build_llm() -> BaseChatModel:
    from langchain_litellm import ChatLiteLLM

    s = get_settings()
    kwargs: dict[str, Any] = {"model": s.llm_model, "temperature": s.llm_temperature}
    if s.llm_api_base:
        kwargs["api_base"] = s.llm_api_base
    return ChatLiteLLM(**kwargs)


def build_graph(llm: BaseChatModel | None = None, checkpointer: BaseCheckpointSaver | None = None):
    llm_with_tools = (llm or build_llm()).bind_tools(ALL_TOOLS)
    tools_by_name = {t.name: t for t in ALL_TOOLS}

    def agent(state: State) -> dict:
        threshold = get_settings().refund_approval_threshold
        system = SystemMessage(SYSTEM_PROMPT + f"\nApproval threshold: refunds over ${threshold:.2f}.")
        return {"messages": [llm_with_tools.invoke([system, *state["messages"]])]}

    def route_after_agent(state: State) -> str:
        last = state["messages"][-1]
        return "tools" if isinstance(last, AIMessage) and last.tool_calls else END

    def tools(state: State, config: RunnableConfig) -> dict:
        """Run the tool calls of the last AI message.

        Human approval is collected FIRST (before any tool has side effects), because a
        resumed interrupt re-executes this node from the top.
        """
        threshold = get_settings().refund_approval_threshold
        customer_id = config["configurable"]["customer_id"]
        calls = state["messages"][-1].tool_calls

        # decisions: tool_call_id -> None (run normally) | ToolMessage (short-circuit result)
        short_circuit: dict[str, ToolMessage] = {}
        for call in calls:
            if call["name"] != REFUND_TOOL_NAME:
                continue
            error, ret = store.preview_refund(customer_id, str(call["args"].get("return_id", "")))
            if error or ret is None:
                short_circuit[call["id"]] = ToolMessage(error or "Refund failed.", tool_call_id=call["id"], name=call["name"])
            elif ret["amount"] > threshold:
                decision = interrupt(
                    {
                        "type": "refund_approval",
                        "return_id": ret["return_id"],
                        "order_id": ret["order_id"],
                        "customer_id": customer_id,
                        "item": ret["name"],
                        "quantity": ret["quantity"],
                        "amount": ret["amount"],
                        "reason": ret["reason"],
                        "threshold": threshold,
                    }
                )
                if not (isinstance(decision, dict) and decision.get("approved") is True):
                    note = decision.get("note", "") if isinstance(decision, dict) else ""
                    short_circuit[call["id"]] = ToolMessage(
                        f"Refund of ${ret['amount']:.2f} for {ret['return_id']} was DECLINED by support. {note}".strip(),
                        tool_call_id=call["id"], name=call["name"],
                    )

        results: list[ToolMessage] = []
        for call in calls:
            if call["id"] in short_circuit:
                results.append(short_circuit[call["id"]])
                continue
            tool = tools_by_name.get(call["name"])
            if tool is None:
                results.append(ToolMessage(f"Unknown tool {call['name']}.", tool_call_id=call["id"], name=call["name"]))
                continue
            try:
                out = tool.invoke(call, config)  # a ToolCall dict -> returns a ToolMessage
                results.append(out)
            except Exception:  # bad/missing args from the model; details stay out of the reply
                results.append(
                    ToolMessage("The tool call was invalid. Check the arguments and try again.",
                                tool_call_id=call["id"], name=call["name"])
                )
        return {"messages": results}

    g = StateGraph(State)
    g.add_node("agent", agent)
    g.add_node("tools", tools)
    g.add_edge(START, "agent")
    g.add_conditional_edges("agent", route_after_agent, {"tools": "tools", END: END})
    g.add_edge("tools", "agent")
    return g.compile(checkpointer=checkpointer or InMemorySaver())
