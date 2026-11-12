"""Interactive CLI chat. Run:  python -m ecommerce_agent.cli --customer C1001

When a large refund needs approval, the CLI plays the role of the support console
and asks you (the "support agent") to approve or decline.
"""

from __future__ import annotations

import argparse
import uuid

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.types import Command

from .graph import build_graph


def _text(msg: AIMessage) -> str:
    content = msg.content
    if isinstance(content, list):
        return "".join(p.get("text", "") if isinstance(p, dict) else str(p) for p in content)
    return str(content)


def _support_console(payload: dict) -> dict:
    print("\n=== SUPPORT APPROVAL REQUIRED ===")
    print(f"Return {payload['return_id']} | Order {payload['order_id']} | Customer {payload['customer_id']}")
    print(f"Item: {payload['quantity']} x {payload['item']} | Reason: {payload['reason']}")
    print(f"Refund amount: ${payload['amount']:.2f} (threshold ${payload['threshold']:.2f})")
    answer = input("Approve refund? [y/N]: ").strip().lower()
    note = "" if answer == "y" else input("Reason for decline (optional): ").strip()
    print("=================================\n")
    return {"approved": answer == "y", "note": note}


def run_turn(graph, user_text: str, config: dict) -> None:
    payload: object = {"messages": [HumanMessage(user_text)]}
    while True:
        result = graph.invoke(payload, config)
        interrupts = result.get("__interrupt__")
        if not interrupts:
            print(f"\nAssistant: {_text(result['messages'][-1])}\n")
            return
        payload = Command(resume=_support_console(interrupts[0].value))


def main() -> None:
    parser = argparse.ArgumentParser(description="E-commerce support agent")
    parser.add_argument("--customer", default="C1001", help="Signed-in customer id (C1001 or C1002 in demo data)")
    args = parser.parse_args()

    graph = build_graph()
    config = {"configurable": {"thread_id": str(uuid.uuid4()), "customer_id": args.customer}}
    print(f"Signed in as {args.customer}. Type 'quit' to exit.\n")
    while True:
        try:
            text = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if text.lower() in {"quit", "exit"}:
            break
        if text:
            run_turn(graph, text, config)


if __name__ == "__main__":
    main()
