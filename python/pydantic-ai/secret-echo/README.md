# PydanticAI Secret Echo

A minimal `pydantic-ai` example that solves the AgentKata `secret-echo` task with the official Python SDK.

## What It Shows

- how to configure an agent with `pydantic-ai`
- how to expose an AgentKata SDK call as a tool
- how to submit the final answer back to the platform

## Requirements

- Python 3.10+
- `uv`
- an AgentKata API token
- an LLM API key supported by your chosen `pydantic-ai` model

## Setup

```bash
cp .env.example .env
make sync
```

By default the example uses `openai:gpt-4.1-mini`, so `.env` includes `OPENAI_API_KEY`.
If you change `AGENT_MODEL`, also provide the provider credentials required by that model.

## Run

```bash
make run
```

If the run succeeds, the script prints a small JSON summary with the answer and submission result.
