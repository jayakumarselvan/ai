"""LLM-callable tools.

The authenticated customer comes from ``config["configurable"]["customer_id"]``
and is NOT a tool argument, so the model can never act for another customer.
"""

from __future__ import annotations

import json

from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool

from . import store

REFUND_TOOL_NAME = "request_refund"


def _customer(config: RunnableConfig) -> str:
    return config["configurable"]["customer_id"]


@tool
def list_my_orders(config: RunnableConfig) -> str:
    """List all orders (id, status, date, total) of the current customer."""
    return json.dumps(store.list_orders(_customer(config)))


@tool
def get_order_details(order_id: str, config: RunnableConfig) -> str:
    """Get full details (status, items, total, dates) of one order, e.g. ORD-1001."""
    order = store.get_order(_customer(config), order_id)
    return json.dumps(order) if order else "Order not found."


@tool
def track_shipment(order_id: str, config: RunnableConfig) -> str:
    """Get shipping/tracking status of an order."""
    order = store.get_order(_customer(config), order_id)
    if not order:
        return "Order not found."
    return json.dumps(
        {"order_id": order["order_id"], "status": order["status"],
         "tracking_number": order["tracking"], "delivered_on": order["delivered"]}
    )


@tool
def cancel_order(order_id: str, config: RunnableConfig) -> str:
    """Cancel an order. Only possible while the order status is 'processing'."""
    return store.cancel_order(_customer(config), order_id)


@tool
def return_product(order_id: str, sku: str, quantity: int, reason: str, config: RunnableConfig) -> str:
    """Create a return for an item of a delivered order (within the return window).
    Returns a return id (RET-xxxx) that is needed to request the refund."""
    return store.create_return(_customer(config), order_id, sku, quantity, reason)


@tool
def get_return_status(return_id: str, config: RunnableConfig) -> str:
    """Get the status of a return/refund, e.g. RET-0001."""
    ret = store.get_return(_customer(config), return_id)
    return json.dumps(ret) if ret else "Return not found."


@tool(REFUND_TOOL_NAME)
def request_refund(return_id: str, config: RunnableConfig) -> str:
    """Refund a previously created return to the customer's original payment method.
    Refunds above the approval threshold are held for human support approval automatically."""
    return store.execute_refund(_customer(config), return_id)


SAFE_TOOLS = [list_my_orders, get_order_details, track_shipment, cancel_order, return_product, get_return_status]
ALL_TOOLS = [*SAFE_TOOLS, request_refund]
