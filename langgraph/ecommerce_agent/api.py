"""FastAPI service: customer chat + support approval queue for large refunds.

Customer endpoints  (header ``Authorization: Bearer <customer token>``)
  POST /chat                 send a message; may return status "pending_approval"
  GET  /chat/{thread_id}     poll the conversation state / latest reply

Support endpoints   (header ``Authorization: Bearer <support key>``)
  GET  /support/approvals                       list refunds waiting for a decision
  POST /support/approvals/{thread_id}/decision  approve / decline -> resumes the chat

Auth config (env): CUSTOMER_TOKENS="token1:C1001,token2:C1002"  SUPPORT_API_KEY="..."
Replace both with your real identity provider (OIDC/SAML) in production.
"""

from __future__ import annotations

import logging
import os
import secrets
import threading
import uuid
from pathlib import Path
from typing import Literal

from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.responses import FileResponse
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.types import Command
from pydantic import BaseModel, Field

from .graph import build_graph

log = logging.getLogger("ecommerce_agent.api")


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=2000)
    thread_id: str | None = Field(default=None, max_length=64)


class ChatResponse(BaseModel):
    thread_id: str
    status: Literal["completed", "pending_approval"]
    reply: str | None = None


class ApprovalItem(BaseModel):
    thread_id: str
    customer_id: str
    return_id: str
    order_id: str
    item: str
    quantity: int
    amount: float
    reason: str


class DecisionRequest(BaseModel):
    approved: bool
    note: str = Field(default="", max_length=500)


def _text(msg: AIMessage) -> str:
    c = msg.content
    if isinstance(c, list):
        return "".join(p.get("text", "") if isinstance(p, dict) else str(p) for p in c)
    return str(c)


def _bearer(authorization: str | None) -> str:
    if not authorization or not authorization.lower().startswith("bearer "):
        raise HTTPException(401, "Missing bearer token")
    return authorization[7:].strip()


def create_app(graph=None, customer_tokens: dict[str, str] | None = None, support_key: str | None = None) -> FastAPI:
    graph = graph or build_graph()
    if customer_tokens is None:
        pairs = [p.split(":", 1) for p in os.environ.get("CUSTOMER_TOKENS", "").split(",") if ":" in p]
        customer_tokens = {tok.strip(): cid.strip() for tok, cid in pairs}
    support_key = support_key if support_key is not None else os.environ.get("SUPPORT_API_KEY", "")

    app = FastAPI(title="E-commerce support agent")
    owners: dict[str, str] = {}              # thread_id -> customer_id
    pending: dict[str, dict] = {}            # thread_id -> approval payload awaiting support
    last_reply: dict[str, str] = {}          # thread_id -> latest assistant reply
    locks: dict[str, threading.Lock] = {}
    registry_lock = threading.Lock()

    def customer_id(authorization: str | None = Header(default=None)) -> str:
        token = _bearer(authorization)
        for known, cid in customer_tokens.items():
            if secrets.compare_digest(token, known):
                return cid
        raise HTTPException(401, "Invalid token")

    def support_agent(authorization: str | None = Header(default=None)) -> None:
        token = _bearer(authorization)
        if not support_key or not secrets.compare_digest(token, support_key):
            raise HTTPException(401, "Invalid token")

    def thread_lock(thread_id: str) -> threading.Lock:
        with registry_lock:
            return locks.setdefault(thread_id, threading.Lock())

    def run(thread_id: str, cid: str, payload) -> ChatResponse:
        config = {"configurable": {"thread_id": thread_id, "customer_id": cid}}
        try:
            result = graph.invoke(payload, config)
        except Exception:
            log.exception("graph run failed", extra={"thread_id": thread_id})
            raise HTTPException(502, "The assistant is temporarily unavailable. Please try again.")
        interrupts = result.get("__interrupt__")
        if interrupts:
            pending[thread_id] = {"customer_id": cid, **interrupts[0].value}
            log.info("refund awaiting approval", extra={"thread_id": thread_id})
            return ChatResponse(thread_id=thread_id, status="pending_approval",
                                reply="Your refund needs approval from our support team. We'll update you shortly.")
        pending.pop(thread_id, None)
        reply = _text(result["messages"][-1])
        last_reply[thread_id] = reply
        return ChatResponse(thread_id=thread_id, status="completed", reply=reply)

    @app.get("/", include_in_schema=False)
    def test_ui():
        return FileResponse(Path(__file__).parent / "static" / "index.html")

    @app.post("/chat", response_model=ChatResponse)
    def chat(req: ChatRequest, cid: str = Depends(customer_id)):
        thread_id = req.thread_id or uuid.uuid4().hex
        owner = owners.setdefault(thread_id, cid)
        if owner != cid:
            raise HTTPException(404, "Conversation not found")
        with thread_lock(thread_id):
            if thread_id in pending:
                raise HTTPException(409, "This conversation is waiting for support approval")
            return run(thread_id, cid, {"messages": [HumanMessage(req.message)]})

    @app.get("/chat/{thread_id}", response_model=ChatResponse)
    def chat_status(thread_id: str, cid: str = Depends(customer_id)):
        if owners.get(thread_id) != cid:
            raise HTTPException(404, "Conversation not found")
        if thread_id in pending:
            return ChatResponse(thread_id=thread_id, status="pending_approval",
                                reply="Your refund needs approval from our support team.")
        return ChatResponse(thread_id=thread_id, status="completed", reply=last_reply.get(thread_id))

    @app.get("/support/approvals", response_model=list[ApprovalItem], dependencies=[Depends(support_agent)])
    def list_approvals():
        return [ApprovalItem(thread_id=t, **{k: v for k, v in p.items() if k in ApprovalItem.model_fields})
                for t, p in pending.items()]

    @app.post("/support/approvals/{thread_id}/decision", response_model=ChatResponse,
              dependencies=[Depends(support_agent)])
    def decide(thread_id: str, body: DecisionRequest):
        with thread_lock(thread_id):
            item = pending.get(thread_id)
            if not item:
                raise HTTPException(404, "No pending approval for this conversation")
            log.info("refund decision", extra={"thread_id": thread_id, "approved": body.approved})
            return run(thread_id, item["customer_id"],
                       Command(resume={"approved": body.approved, "note": body.note}))

    return app


def app_factory() -> FastAPI:
    """Entry point for ``uvicorn ecommerce_agent.api:app_factory --factory``."""
    return create_app()
