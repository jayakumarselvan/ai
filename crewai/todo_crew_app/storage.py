"""
storage.py
----------
Simple JSON-file-backed persistence layer for the todo app.
No external DB needed — everything lives in data/todos.json.

A file lock (filelock) is used so the FastAPI server (which may handle
concurrent requests) doesn't corrupt the JSON file with interleaved writes.
"""
from __future__ import annotations

import json
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import Any

DATA_DIR = Path(os.getenv("TODO_DATA_DIR", Path(__file__).parent / "data"))
DATA_FILE = DATA_DIR / "todos.json"

_lock = Lock()  # in-process guard; fine for a single uvicorn worker


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _ensure_store() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if not DATA_FILE.exists():
        DATA_FILE.write_text(json.dumps({"todos": []}, indent=2))


def _read() -> dict[str, Any]:
    _ensure_store()
    with open(DATA_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def _write(data: dict[str, Any]) -> None:
    tmp_file = DATA_FILE.with_suffix(".tmp")
    with open(tmp_file, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    tmp_file.replace(DATA_FILE)  # atomic on POSIX


def list_todos(status: str | None = None) -> list[dict[str, Any]]:
    with _lock:
        todos = _read()["todos"]
    if status:
        todos = [t for t in todos if t["status"] == status]
    return todos


def get_todo(todo_id: str) -> dict[str, Any] | None:
    with _lock:
        todos = _read()["todos"]
    return next((t for t in todos if t["id"] == todo_id), None)


def add_todo(title: str, description: str = "") -> dict[str, Any]:
    with _lock:
        data = _read()
        todo = {
            "id": uuid.uuid4().hex[:8],
            "title": title,
            "description": description,
            "status": "pending",
            "created_at": _now(),
            "updated_at": _now(),
        }
        data["todos"].append(todo)
        _write(data)
    return todo


def update_todo(
    todo_id: str,
    title: str | None = None,
    description: str | None = None,
    status: str | None = None,
) -> dict[str, Any] | None:
    with _lock:
        data = _read()
        for t in data["todos"]:
            if t["id"] == todo_id:
                if title is not None:
                    t["title"] = title
                if description is not None:
                    t["description"] = description
                if status is not None:
                    t["status"] = status
                t["updated_at"] = _now()
                _write(data)
                return t
    return None


def complete_todo(todo_id: str) -> dict[str, Any] | None:
    return update_todo(todo_id, status="completed")


def delete_todo(todo_id: str) -> bool:
    with _lock:
        data = _read()
        before = len(data["todos"])
        data["todos"] = [t for t in data["todos"] if t["id"] != todo_id]
        deleted = len(data["todos"]) != before
        if deleted:
            _write(data)
    return deleted
