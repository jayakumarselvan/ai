# mcp-stateless-demo

A small demo of a **stateless MCP server** (MCP Python SDK 2.x) and a client that lets any LLM call its tools through [LiteLLM](https://docs.litellm.ai/).

The server exposes one tool, `get_order_status`. The client asks an LLM a question, the LLM calls the tool over MCP, and the answer comes back.

## How it works

- `server.py`: an MCP server running over streamable HTTP in **stateless mode**, so it keeps no session between requests.
- `client.py`: connects with a fresh MCP connection for each question, passes the server's tools to the LLM via LiteLLM, and runs the tool calls.

## Project structure

```
mcp-stateless-demo/
├── server.py
├── client.py
├── requirements.txt
├── .env.sample
├── .gitignore
└── README.md
```

## Setup

Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
```

Install the dependencies:

```bash
pip install -r requirements.txt
```

`requirements.txt`:

```
mcp==2.3.0
litellm==1.108.0
python-dotenv
```

## Configuration

Copy the sample file to `.env` (`cp .env.sample .env`) and add your key, or create `.env` yourself in the project folder:

```ini
GROQ_API_KEY=your-groq-key-here
MODEL=groq/openai/gpt-oss-120b
MCP_URL=http://localhost:8000/mcp
```

- `GROQ_API_KEY`: get one from the [Groq console](https://console.groq.com).
- `MODEL`: any [LiteLLM](https://docs.litellm.ai/docs/providers) model string. Pick a model that supports tool calling.
- `MCP_URL`: where the server listens.

To use another provider, change `MODEL` and set that provider's API key (for example `OPENAI_API_KEY` for `gpt-4o-mini`).

Add `.env` to `.gitignore` so your key is not committed.

## Run

Start the server in one terminal:

```bash
python server.py
```

Run the client in a second terminal:

```bash
python client.py
```

The client asks about orders `1001` and `1002` and prints the answers.

Result

Order #1001 has already been shipped. It’s on its way and is expected to arrive on **November 8, 2026**.

Your order (ID 1002) is currently **delayed**. The updated estimated delivery date is **November 12, 2026**. If you need any further assistance or would like to explore alternatives (e.g., expedited shipping, a replacement, or a refund), just let me know!