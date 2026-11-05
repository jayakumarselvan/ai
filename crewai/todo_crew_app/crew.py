"""
crew.py
-------
Defines the single-agent CrewAI "todo assistant" crew.

CrewAI's `LLM` class is a thin wrapper around LiteLLM, so pointing it at
OpenAI just means using a LiteLLM-style model string ("openai/gpt-4o-mini")
and letting LiteLLM pick up OPENAI_API_KEY from the environment.

Swapping providers later (e.g. watsonx / IBM Granite via LiteLLM) only
means changing MODEL_NAME + the matching provider env vars — nothing else
in this file needs to change.
"""
from __future__ import annotations

import os

from crewai import Agent, Crew, Process, Task
from crewai import LLM
from dotenv import load_dotenv

from tools import get_all_tools

load_dotenv()

# LiteLLM-style model string. Examples:
#   "openai/gpt-4o-mini"          (OpenAI)
#   "watsonx/ibm/granite-3-8b-instruct"  (watsonx via LiteLLM, for later)
MODEL_NAME = os.getenv("TODO_LLM_MODEL", "openai/gpt-4o-mini")


def build_llm() -> LLM:
    return LLM(
        model=MODEL_NAME,
        api_key=os.getenv("OPENAI_API_KEY"),
        temperature=0.2,
    )


def build_agent() -> Agent:
    return Agent(
        role="Todo List Assistant",
        goal=(
            "Help the user manage their todo list accurately by adding, "
            "listing, updating, completing, and deleting items using the "
            "available tools. Always use a tool for any action that reads "
            "or changes the todo list — never invent todo data yourself."
        ),
        backstory=(
            "You are a precise, no-nonsense productivity assistant embedded "
            "in a chat interface. Users will talk to you in natural language "
            "and expect their todo list to be updated correctly and "
            "confirmed back to them in a short, clear reply."
        ),
        tools=get_all_tools(),
        llm=build_llm(),
        verbose=False,
        allow_delegation=False,
    )


def build_task(agent: Agent, user_message: str) -> Task:
    return Task(
        description=(
            "The user said:\n\n"
            f'"{user_message}"\n\n'
            "Figure out what todo-list action(s) they want (add, list, "
            "update, complete, or delete) and use the correct tool(s) to "
            "perform them. If they're just asking a question about their "
            "list, use the list_todos tool to check before answering."
        ),
        expected_output=(
            "A short, friendly natural-language confirmation of what was "
            "done (or the requested information), written for a chat UI. "
            "Do not include raw JSON in the final answer unless the user "
            "asked for it."
        ),
        agent=agent,
    )


def run_todo_crew(user_message: str) -> str:
    """Run the crew once for a single user message and return the reply text."""
    agent = build_agent()
    task = build_task(agent, user_message)
    crew = Crew(
        agents=[agent],
        tasks=[task],
        process=Process.sequential,
        verbose=False,
    )
    result = crew.kickoff()
    # CrewOutput.raw contains the plain-text final answer.
    # Fall back to str() only if .raw is absent (older crewai versions).
    return (result.raw or str(result)).strip()
