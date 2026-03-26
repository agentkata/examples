# Google ADK Conditional Path

A Go example that solves the AgentKata `conditional-path` task using a Gemini-powered LLM agent with Google ADK.

## What It Shows

- how to build a reactive LLM agent with Google ADK and function tools
- how Gemini reasons through a multi-step branching workflow using tools
- how the LLM handles transient API errors (retries) and local computation

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

The agent fetches the workflow config, calls API/lookup tools as needed, computes the final token, and prints a JSON summary with the answer and submission result.
