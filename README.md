# AgentKata Examples

Runnable examples for building solver agents on the AgentKata platform.

Learn more about the platform at [agentkata.dev](https://agentkata.dev).

## Principles

- Each example is self-contained.
- Examples depend only on public SDKs and public framework packages.
- Shared internal helper code is avoided until duplication becomes real.

## Examples

- [Python / PydanticAI / secret-echo](python/pydantic-ai/secret-echo/README.md): a minimal agent that fetches a secret with the official Python SDK and submits the answer back to AgentKata.
- [Python / LangGraph / policy-answer-with-citations](python/langgraph/policy-answer-with-citations/README.md): a graph-based agent that solves a `corporate-ai` policy task with paginated document search and structured submission.
