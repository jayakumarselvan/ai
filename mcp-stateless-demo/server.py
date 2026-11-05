from mcp.server import MCPServer

mcp = MCPServer("orders", version="1.0.0")

ORDERS = {
    "1001": {"status": "Shipped", "eta": "2026-11-08"},
    "1002": {"status": "Delayed", "eta": "2026-11-12"},
}


@mcp.tool()
def get_order_status(order_id: str) -> dict:
    """Look up the status of a customer order by its ID."""
    return ORDERS.get(order_id, {"error": "Order not found"})


if __name__ == "__main__":
    # In mcp 2.x, transport options are passed to run(), not the constructor.
    # stateless_http=True: no session is kept between requests.
    mcp.run(
        transport="streamable-http",
        host="127.0.0.1",
        port=8000,
        stateless_http=True,
        json_response=True,
    )  # serves at http://localhost:8000/mcp
