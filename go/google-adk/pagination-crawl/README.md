# Google ADK Pagination Crawl

A Go example that solves the AgentKata `pagination-crawl` task using a Gemini-powered LLM agent with Google ADK.

## What It Shows

- how to build a reactive LLM agent with Google ADK and function tools
- how Gemini crawls paginated data autonomously using a single tool
- how the LLM handles transient errors and tracks state across tool calls

## Requirements

- Go 1.25+
- an AgentKata API token
- a Google API key (Gemini)

## Setup

```bash
cp .env.example .env
# fill in AGENTKATA_API_TOKEN and GOOGLE_API_KEY
make sync
```

## Run

```bash
make run
```

The agent pages through all items, finds the one with the highest value, and prints a JSON summary with the winning item ID and submission result.
