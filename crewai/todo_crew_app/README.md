# Todo CrewAI Agent

## Setup

```bash
cd todo_crew_app
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # then edit .env and set OPENAI_API_KEY
```

## Run

```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

- Swagger UI: `http://localhost:8000/docs`
- Health check: `GET /health`

## Try it like watsonx Orchestrate would

```bash
curl -X POST http://localhost:8000/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
        "model": "todo-crew-agent",
        "messages": [
          {"role": "user", "content": "Add a todo to buy milk tomorrow"}
        ]
      }'
```

Response shape:

```json
{
  "id": "chatcmpl-xxxxxxxx",
  "object": "chat.completion",
  "created": 1737000000,
  "model": "todo-crew-agent",
  "choices": [
    {
      "index": 0,
      "message": {"role": "assistant", "content": "Added 'buy milk tomorrow' to your todo list."},
      "finish_reason": "stop"
    }
  ],
  "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
}
```

Try a few more once that works:

```bash
curl -X POST http://localhost:8000/chat/completions -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"What'\''s on my todo list?"}]}'

curl -X POST http://localhost:8000/chat/completions -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"Mark buy milk as done"}]}'
```

## Registering with watsonx Orchestrate

Register this service as an **OpenAI-compatible external chat agent**,
pointing it at:

```
https://<your-host>/chat/completions
```

Orchestrate will POST `{model, messages}` and expects back the standard
`chat.completion` object shown above — which is exactly what this endpoint
returns. If you host this behind Hostinger/another provider, make sure the
route is reachable over HTTPS and not behind auth Orchestrate can't pass
through (add an API-key check in `main.py` if you need one — none is
enforced by default here).

## Swapping the LLM provider later

Since the agent goes through CrewAI's `LLM` (a LiteLLM wrapper), switching
providers is just an env var change in `crew.py` / `.env` — e.g. to route
through watsonx instead of OpenAI directly:

```
TODO_LLM_MODEL=watsonx/ibm/granite-3-8b-instruct
WATSONX_URL=...
WATSONX_APIKEY=...
WATSONX_PROJECT_ID=...
```

No other file needs to change.

## Data storage

Todos live in `data/todos.json`:

```json
{
  "todos": [
    {
      "id": "a1b2c3d4",
      "title": "Buy milk",
      "description": "",
      "status": "pending",
      "created_at": "2026-07-16T09:00:00+00:00",
      "updated_at": "2026-07-16T09:00:00+00:00"
    }
  ]
}
```

Writes are atomic (write to a temp file, then rename) so a crash mid-write
won't corrupt the file.
