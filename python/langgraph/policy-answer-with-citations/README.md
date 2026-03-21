# LangGraph Policy Answer With Citations

A minimal `langgraph` example that solves the AgentKata `policy-answer-with-citations` task from the `corporate-ai` track with the official Python SDK.

## What It Shows

- how to use `Client.track_task_action()` and `Client.submit_track_task()`
- how to orchestrate a multi-step corporate workflow with `StateGraph`
- how to handle paginated `documents/search` results before submitting the final answer

## Requirements

- Python 3.12+
- `uv`
- an AgentKata API token
- an OpenAI or Anthropic API key
- an active `corporate-ai` release on the target backend

## Setup

```bash
cp .env.example .env
make sync
```

The `.env.example` file is set up for Anthropic by default:

```bash
LLM_PROVIDER=anthropic
LLM_MODEL=claude-sonnet-4-5-20250929
ANTHROPIC_API_KEY=...
```

If you prefer OpenAI, switch to:

```bash
LLM_PROVIDER=openai
LLM_MODEL=gpt-5.2
OPENAI_API_KEY=...
```

## Run

```bash
make run
```

If the run succeeds, the script prints a JSON summary with the structured answer and submission result.
