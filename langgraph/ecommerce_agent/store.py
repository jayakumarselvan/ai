"""In-memory mock of the order system (replace with real DB / service calls).

Every function takes the authenticated ``customer_id`` and enforces ownership,
so a customer can never read or change another customer's order.
"""

from __future__ import annotations

import copy
import itertools
from datetime import datetime, timedelta, timezone

from .config import get_settings

_NOW = datetime.now(timezone.utc)


def _days_ago(n: int) -> str:
    return (_NOW - timedelta(days=n)).date().isoformat()


def _seed_orders() -> dict[str, dict]:
    return {
        "ORD-1001": {
            "customer_id": "C1001", "status": "delivered", "placed": _days_ago(8),
            "delivered": _days_ago(5), "tracking": "TRK-9001",
            "items": [
                {"sku": "HP-100", "name": "Wireless Headphones", "price": 59.99, "qty": 1, "returned_qty": 0},
                {"sku": "CB-USBC", "name": "USB-C Cable", "price": 12.99, "qty": 2, "returned_qty": 0},
            ],
        },
        "ORD-1002": {
            "customer_id": "C1001", "status": "delivered", "placed": _days_ago(14),
            "delivered": _days_ago(10), "tracking": "TRK-9002",
            "items": [
                {"sku": "KB-MECH", "name": "Mechanical Keyboard", "price": 149.00, "qty": 1, "returned_qty": 0},
            ],
        },
        "ORD-1003": {
            "customer_id": "C1001", "status": "shipped", "placed": _days_ago(2),
            "delivered": None, "tracking": "TRK-9003",
            "items": [
                {"sku": "MS-WL", "name": "Wireless Mouse", "price": 29.50, "qty": 1, "returned_qty": 0},
            ],
        },
        "ORD-1004": {
            "customer_id": "C1001", "status": "processing", "placed": _days_ago(0),
            "delivered": None, "tracking": None,
            "items": [
                {"sku": "MON-27", "name": "27-inch Monitor", "price": 219.00, "qty": 1, "returned_qty": 0},
            ],
        },
        "ORD-1005": {
            "customer_id": "C1001", "status": "delivered", "placed": _days_ago(65),
            "delivered": _days_ago(60), "tracking": "TRK-9005",
            "items": [
                {"sku": "WC-HD", "name": "HD Webcam", "price": 45.00, "qty": 1, "returned_qty": 0},
            ],
        },
        "ORD-2001": {
            "customer_id": "C1002", "status": "delivered", "placed": _days_ago(6),
            "delivered": _days_ago(3), "tracking": "TRK-9101",
            "items": [
                {"sku": "SP-BT", "name": "Bluetooth Speaker", "price": 89.00, "qty": 1, "returned_qty": 0},
            ],
        },
    }


ORDERS: dict[str, dict] = {}
RETURNS: dict[str, dict] = {}
_return_ids = itertools.count(1)


def reset() -> None:
    """Restore seed data (used by tests and at start-up)."""
    global _return_ids
    ORDERS.clear()
    ORDERS.update(copy.deepcopy(_seed_orders()))
    RETURNS.clear()
    _return_ids = itertools.count(1)


reset()


def _order_total(order: dict) -> float:
    return round(sum(i["price"] * i["qty"] for i in order["items"]), 2)


def get_order(customer_id: str, order_id: str) -> dict | None:
    order = ORDERS.get(order_id.strip().upper())
    # Same answer for "missing" and "not yours" so IDs cannot be probed.
    if not order or order["customer_id"] != customer_id:
        return None
    return {"order_id": order_id.strip().upper(), "total": _order_total(order), **copy.deepcopy(order)}


def list_orders(customer_id: str) -> list[dict]:
    return [
        {"order_id": oid, "status": o["status"], "placed": o["placed"], "total": _order_total(o)}
        for oid, o in ORDERS.items()
        if o["customer_id"] == customer_id
    ]


def cancel_order(customer_id: str, order_id: str) -> str:
    order = get_order(customer_id, order_id)
    if not order:
        return "Order not found."
    if order["status"] != "processing":
        return f"Order {order['order_id']} is '{order['status']}' and can no longer be cancelled."
    ORDERS[order["order_id"]]["status"] = "cancelled"
    return f"Order {order['order_id']} has been cancelled."


def create_return(customer_id: str, order_id: str, sku: str, quantity: int, reason: str) -> str:
    order = get_order(customer_id, order_id)
    if not order:
        return "Order not found."
    if order["status"] != "delivered":
        return f"Order {order['order_id']} is '{order['status']}'; only delivered orders can be returned."
    window = get_settings().return_window_days
    delivered = datetime.fromisoformat(order["delivered"]).replace(tzinfo=timezone.utc)
    if (_NOW - delivered).days > window:
        return f"The {window}-day return window for order {order['order_id']} has expired."
    item = next((i for i in ORDERS[order["order_id"]]["items"] if i["sku"].upper() == sku.strip().upper()), None)
    if not item:
        return f"SKU {sku} is not part of order {order['order_id']}."
    if quantity < 1 or quantity > item["qty"] - item["returned_qty"]:
        return f"Invalid quantity. Returnable quantity for {item['sku']} is {item['qty'] - item['returned_qty']}."
    item["returned_qty"] += quantity
    return_id = f"RET-{next(_return_ids):04d}"
    RETURNS[return_id] = {
        "customer_id": customer_id, "order_id": order["order_id"], "sku": item["sku"],
        "name": item["name"], "quantity": quantity, "amount": round(item["price"] * quantity, 2),
        "reason": reason, "status": "requested",
    }
    return (
        f"Return {return_id} created for {quantity} x {item['name']} "
        f"(refundable amount ${RETURNS[return_id]['amount']:.2f})."
    )


def get_return(customer_id: str, return_id: str) -> dict | None:
    ret = RETURNS.get(return_id.strip().upper())
    if not ret or ret["customer_id"] != customer_id:
        return None
    return {"return_id": return_id.strip().upper(), **ret}


def preview_refund(customer_id: str, return_id: str) -> tuple[str | None, dict | None]:
    """Validate a refund without executing it. Returns (error, return_record).

    The amount always comes from the system of record, never from the LLM.
    """
    ret = get_return(customer_id, return_id)
    if not ret:
        return "Return not found. A return must be created before a refund can be requested.", None
    if ret["status"] == "refunded":
        return f"Return {ret['return_id']} has already been refunded.", None
    return None, ret


def execute_refund(customer_id: str, return_id: str) -> str:
    error, ret = preview_refund(customer_id, return_id)
    if error or ret is None:
        return error or "Refund failed."
    RETURNS[ret["return_id"]]["status"] = "refunded"
    return f"Refund of ${ret['amount']:.2f} for return {ret['return_id']} was issued to the original payment method."
