"""
main.py
-------
FastAPI server exposing an OpenAI-compatible `/chat/completions` endpoint
backed by a CrewAI agent. This is the same shape watsonx Orchestrate expects
when you register this service as an "external agent" (chat-completions
style): POST a `messages` array in, get back a standard
`chat.completion` object.

Run locally:
    uvicorn main:app --host 0.0.0.0 --port 8000

Then register http://<host>:8000/chat/completions with watsonx Orchestrate
as an OpenAI-compatible external agent.
"""

from __future__ import annotations

import time
import uuid
from typing import Literal

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

import storage
from crew import run_todo_crew

app = FastAPI(
    title="Todo CrewAI Agent",
    description=(
        "OpenAI-compatible chat-completions API backed by a CrewAI agent "
        "that manages a JSON-file todo list. Suitable for registration as "
        "an external agent in watsonx Orchestrate."
    ),
    version="1.0.0",
)


# ---------------------------------------------------------------------------
# OpenAI-compatible schema (subset needed for watsonx Orchestrate / clients
# that speak the chat.completions protocol)
# ---------------------------------------------------------------------------
class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant", "tool"]
    content: str


class ChatCompletionRequest(BaseModel):
    model: str = "todo-crew-agent"
    messages: list[ChatMessage]
    stream: bool = False
    temperature: float | None = None
    max_tokens: int | None = None


class ChatCompletionChoice(BaseModel):
    index: int
    message: ChatMessage
    finish_reason: str = "stop"


class Usage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: list[ChatCompletionChoice]
    usage: Usage = Field(default_factory=Usage)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/chat/completions", response_model=ChatCompletionResponse)
def chat_completions(req: ChatCompletionRequest) -> ChatCompletionResponse:
    # watsonx Orchestrate sends stream=true by default; we ignore it and always
    # return a standard non-streaming chat.completion response.

    user_messages = [m for m in req.messages if m.role == "user"]
    if not user_messages:
        raise HTTPException(
            status_code=400, detail="No user message found in 'messages'."
        )
    last_user_message = user_messages[-1].content

    try:
        reply_text = run_todo_crew(last_user_message)
    except Exception as exc:  # surface crew/LLM errors as a clean 500
        raise HTTPException(status_code=500, detail=f"Agent error: {exc}") from exc

    # watsonx Orchestrate raises "No generation chunks were returned" when
    # content is empty — ensure there is always something to return.
    if not reply_text:
        reply_text = "Done."

    return ChatCompletionResponse(
        id=f"chatcmpl-{uuid.uuid4().hex[:12]}",
        created=int(time.time()),
        model=req.model,
        choices=[
            ChatCompletionChoice(
                index=0,
                message=ChatMessage(role="assistant", content=reply_text),
            )
        ],
    )


# ---------------------------------------------------------------------------
# Plain REST endpoints for the todo list itself (handy for a UI or debugging,
# independent of the chat-completions path)
# ---------------------------------------------------------------------------
class TodoIn(BaseModel):
    title: str
    description: str = ""


class TodoUpdateIn(BaseModel):
    title: str | None = None
    description: str | None = None
    status: str | None = None


@app.get("/todos")
def get_todos(status: str | None = None) -> list[dict]:
    return storage.list_todos(status=status)


@app.post("/todos")
def create_todo(todo: TodoIn) -> dict:
    return storage.add_todo(title=todo.title, description=todo.description)


@app.get("/todos/{todo_id}")
def get_todo(todo_id: str) -> dict:
    todo = storage.get_todo(todo_id)
    if not todo:
        raise HTTPException(status_code=404, detail="Todo not found")
    return todo


@app.patch("/todos/{todo_id}")
def patch_todo(todo_id: str, todo: TodoUpdateIn) -> dict:
    updated = storage.update_todo(
        todo_id, title=todo.title, description=todo.description, status=todo.status
    )
    if not updated:
        raise HTTPException(status_code=404, detail="Todo not found")
    return updated


@app.delete("/todos/{todo_id}")
def remove_todo(todo_id: str) -> dict:
    ok = storage.delete_todo(todo_id)
    if not ok:
        raise HTTPException(status_code=404, detail="Todo not found")
    return {"deleted": todo_id}
