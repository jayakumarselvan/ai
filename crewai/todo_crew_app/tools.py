"""
tools.py
--------
CrewAI tools that let the agent operate on the JSON-file todo store.
Each tool is a small, single-purpose BaseTool subclass so the LLM
(via function calling / tool use) can pick the right operation.
"""

from __future__ import annotations

import json

from crewai.tools import BaseTool
from pydantic import BaseModel, Field

import storage


class AddTodoInput(BaseModel):
    # NOTE: field is named "task_title", not "title" -- OpenAI's strict-mode
    # schema sanitizer treats "title" as a reserved JSON-Schema metadata
    # keyword and strips any key with that literal name, which corrupts
    # the tool schema if a property is also called "title".
    task_title: str = Field(..., description="Short title of the todo item")
    description: str = Field("", description="Optional longer description")


class AddTodoTool(BaseTool):
    name: str = "add_todo"
    description: str = (
        "Create a new todo item with a title and optional description. "
        "Returns the created todo as JSON."
    )
    args_schema: type[BaseModel] = AddTodoInput

    def _run(self, task_title: str, description: str = "") -> str:
        todo = storage.add_todo(title=task_title, description=description)
        return json.dumps(todo)


class ListTodosInput(BaseModel):
    status: str | None = Field(
        None,
        description="Optional filter: 'pending' or 'completed'. Omit to list all.",
    )


class ListTodosTool(BaseTool):
    name: str = "list_todos"
    description: str = (
        "List todo items, optionally filtered by status "
        "('pending' or 'completed'). Returns a JSON array."
    )
    args_schema: type[BaseModel] = ListTodosInput

    def _run(self, status: str | None = None) -> str:
        todos = storage.list_todos(status=status)
        if not todos:
            return "No todos found."
        return json.dumps(todos)


class CompleteTodoInput(BaseModel):
    todo_id: str = Field(..., description="The id of the todo to mark complete")


class CompleteTodoTool(BaseTool):
    name: str = "complete_todo"
    description: str = "Mark a todo item as completed given its id."
    args_schema: type[BaseModel] = CompleteTodoInput

    def _run(self, todo_id: str) -> str:
        todo = storage.complete_todo(todo_id)
        if not todo:
            return f"No todo found with id '{todo_id}'."
        return json.dumps(todo)


class UpdateTodoInput(BaseModel):
    todo_id: str = Field(..., description="The id of the todo to update")
    # NOTE: "task_title" not "title" -- see comment in AddTodoInput above.
    task_title: str | None = Field(None, description="New title, if changing it")
    description: str | None = Field(None, description="New description, if changing it")
    status: str | None = Field(None, description="New status: 'pending' or 'completed'")


class UpdateTodoTool(BaseTool):
    name: str = "update_todo"
    description: str = (
        "Update the title, description, and/or status of an existing todo, "
        "identified by its id. Only the fields provided are changed."
    )
    args_schema: type[BaseModel] = UpdateTodoInput

    def _run(
        self,
        todo_id: str,
        task_title: str | None = None,
        description: str | None = None,
        status: str | None = None,
    ) -> str:
        todo = storage.update_todo(
            todo_id, title=task_title, description=description, status=status
        )
        if not todo:
            return f"No todo found with id '{todo_id}'."
        return json.dumps(todo)


class DeleteTodoInput(BaseModel):
    todo_id: str = Field(..., description="The id of the todo to delete")


class DeleteTodoTool(BaseTool):
    name: str = "delete_todo"
    description: str = "Permanently delete a todo item given its id."
    args_schema: type[BaseModel] = DeleteTodoInput

    def _run(self, todo_id: str) -> str:
        ok = storage.delete_todo(todo_id)
        return (
            f"Deleted todo '{todo_id}'."
            if ok
            else f"No todo found with id '{todo_id}'."
        )


def get_all_tools() -> list[BaseTool]:
    return [
        AddTodoTool(),
        ListTodosTool(),
        CompleteTodoTool(),
        UpdateTodoTool(),
        DeleteTodoTool(),
    ]
