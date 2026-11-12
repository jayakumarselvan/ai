# E-commerce support agent (LangGraph + LiteLLM)

Customers chat to: view orders, track shipments, cancel orders, return products, and get refunds.
Refunds **above $100** pause the graph and wait for a human support agent to approve or decline.

## Flow

```
START -> agent (LLM + tools bound) -> tools -> agent -> ... -> END
                                       |
                    request_refund > threshold? -> interrupt() -> support approves/declines
```

- `ecommerce_agent/graph.py` – graph, system prompt, approval gate (`interrupt` / `Command(resume=...)`)
- `ecommerce_agent/tools.py` – tools; the customer id comes from graph config, not from the LLM
- `ecommerce_agent/store.py` – mock order system (swap for real DB/APIs); refund amount is always read from here, never from the LLM
- `ecommerce_agent/api.py` – FastAPI service: customer chat + support approval queue
- `ecommerce_agent/cli.py` – chat CLI that also acts as the support console

## Run

```bash
python3.13 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env     # set LLM_MODEL (+ API key) - any LiteLLM model
python -m ecommerce_agent.cli --customer C1001
pytest                   # tests use a fake LLM, no API key needed
```

## API (FastAPI)

```bash
export CUSTOMER_TOKENS="$(python -c 'import secrets;print(secrets.token_urlsafe(32))'):C1001"
export SUPPORT_API_KEY="$(python -c 'import secrets;print(secrets.token_urlsafe(32))')"
uvicorn ecommerce_agent.api:app_factory --factory --host 127.0.0.1 --port 8000
```

| Who | Endpoint | Purpose |
|---|---|---|
| Customer | `POST /chat` `{"message", "thread_id?"}` | Chat. A refund > threshold returns `status: "pending_approval"` |
| Customer | `GET /chat/{thread_id}` | Poll; shows the final reply once support decides |
| Support | `GET /support/approvals` | Refunds waiting for a decision |
| Support | `POST /support/approvals/{thread_id}/decision` `{"approved": true, "note": ""}` | Approve/decline; resumes the chat and issues the refund |

All calls need `Authorization: Bearer <token>`. Customer tokens map to a customer id; the support key is separate.

Demo orders for C1001: `ORD-1001` (delivered, small items), `ORD-1002` (keyboard $149 -> needs approval),
`ORD-1003` (shipped), `ORD-1004` (processing, cancellable), `ORD-1005` (outside return window).

Example: *"Return the keyboard from ORD-1002, it's the wrong one, and refund me"* -> support approval prompt.

## Production notes

- Replace `InMemorySaver` and the API's in-memory pending/owner maps with a database so pending approvals survive restarts and multiple workers.
- In a web deployment, the approval is `graph.invoke(Command(resume={...}), config)` from a support UI/API using the same `thread_id`.
- Take `customer_id` from your authenticated session, never from chat text.
