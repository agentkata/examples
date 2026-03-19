from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any

from agentkata import AgentKataAPIError, Client, RequestMeta
from dotenv import load_dotenv
from pydantic_ai import Agent, RunContext

DEFAULT_MODEL = "openai:gpt-4.1-mini"
DEFAULT_TASK_ID = "secret-echo"
SOLVE_PROMPT = "Solve the AgentKata secret-echo task."


@dataclass(frozen=True)
class Settings:
    base_url: str
    api_token: str
    model: str
    task_id: str


@dataclass(frozen=True)
class SolverDeps:
    client: Client
    task_id: str


def build_agent(model: str) -> Agent[SolverDeps, str]:
    agent = Agent(
        model,
        deps_type=SolverDeps,
        output_type=str,
        instructions=(
            "You are solving the AgentKata task `secret-echo`. "
            "Use the `get_secret` tool to retrieve the secret value. "
            "Return only the secret string, with no extra text, quotes, or formatting."
        ),
    )

    @agent.tool
    def get_secret(ctx: RunContext[SolverDeps]) -> str:
        """Fetch the secret value for the current task."""
        envelope = ctx.deps.client.task_action(task_id=ctx.deps.task_id, action="secret")
        _ensure_success(envelope, operation="get_secret")

        data = envelope.data
        if not isinstance(data, dict):
            raise RuntimeError(f"unexpected secret payload type: {type(data).__name__}")

        secret = data.get("secret")
        if not isinstance(secret, str) or not secret.strip():
            raise RuntimeError("secret payload does not contain a non-empty string")
        return secret.strip()

    return agent


def load_settings() -> Settings:
    load_dotenv()
    return Settings(
        base_url=require_env("AGENTKATA_BASE_URL"),
        api_token=require_env("AGENTKATA_API_TOKEN"),
        model=os.getenv("AGENT_MODEL", DEFAULT_MODEL),
        task_id=os.getenv("AGENTKATA_TASK_ID", DEFAULT_TASK_ID),
    )


def require_env(name: str) -> str:
    value = os.getenv(name, "").strip()
    if value:
        return value
    raise RuntimeError(f"missing required environment variable: {name}")


def _ensure_success(envelope: Any, *, operation: str) -> None:
    error = getattr(envelope, "error", None)
    if error is None:
        return

    code = getattr(error, "code", "") or "UNKNOWN_ERROR"
    message = getattr(error, "message", "request failed") or "request failed"
    raise RuntimeError(f"{operation} failed: {code}: {message}")


def _normalize_answer(answer: str) -> str:
    stripped = answer.strip()
    if len(stripped) >= 2 and stripped[0] == stripped[-1] and stripped[0] in {'"', "'"}:
        return stripped[1:-1].strip()
    return stripped


def _to_plain_data(value: Any) -> Any:
    if value is None:
        return None
    if hasattr(value, "to_dict") and callable(value.to_dict):
        return value.to_dict()
    if hasattr(value, "model_dump") and callable(value.model_dump):
        return value.model_dump(mode="json")
    if isinstance(value, dict):
        return {str(key): _to_plain_data(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_to_plain_data(item) for item in value]
    return value


def main() -> None:
    settings = load_settings()
    agent = build_agent(settings.model)

    with Client(base_url=settings.base_url, api_token=settings.api_token) as client:
        restart = client.restart_task(task_id=settings.task_id)
        _ensure_success(restart, operation="restart_task")

        deps = SolverDeps(client=client, task_id=settings.task_id)
        result = agent.run_sync(SOLVE_PROMPT, deps=deps)
        answer = _normalize_answer(result.output)
        usage = result.usage()

        submit = client.submit_task(
            task_id=settings.task_id,
            answer=answer,
            meta=RequestMeta(
                model=settings.model,
                prompt_tokens=usage.input_tokens or 0,
                completion_tokens=usage.output_tokens or 0,
            ),
        )
        _ensure_success(submit, operation="submit_task")

        summary = {
            "task_id": settings.task_id,
            "model": settings.model,
            "answer": answer,
            "submission": _to_plain_data(submit.data),
        }
        print(json.dumps(summary, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    try:
        main()
    except AgentKataAPIError as exc:
        raise SystemExit(f"AgentKata request failed: {exc.code}: {exc.message}") from exc
    except Exception as exc:
        raise SystemExit(str(exc)) from exc
