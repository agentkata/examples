import json
import os
from dataclasses import dataclass

from agentkata import AgentKataAPIError, Client, RequestMeta
from pydantic_ai import Agent, RunContext
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    agentkata_base_url: str
    agentkata_api_token: str
    agentkata_task_id: str = "secret-echo"
    agent_model: str = "openai:gpt-5-mini"
    openai_api_key: str = ""
    anthropic_api_key: str = ""


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
        return envelope.data["secret"]

    return agent


def main() -> None:
    settings = Settings()
    os.environ.setdefault("OPENAI_API_KEY", settings.openai_api_key)
    os.environ.setdefault("ANTHROPIC_API_KEY", settings.anthropic_api_key)
    agent = build_agent(settings.agent_model)

    with Client(
        base_url=settings.agentkata_base_url, api_token=settings.agentkata_api_token
    ) as client:
        client.restart_task(task_id=settings.agentkata_task_id)

        deps = SolverDeps(client=client, task_id=settings.agentkata_task_id)
        result = agent.run_sync("Solve the AgentKata secret-echo task.", deps=deps)
        usage = result.usage()

        submit = client.submit_task(
            task_id=settings.agentkata_task_id,
            answer=result.output,
            meta=RequestMeta(
                model=settings.agent_model,
                prompt_tokens=usage.input_tokens or 0,
                completion_tokens=usage.output_tokens or 0,
            ),
        )

        print(
            json.dumps(
                {
                    "task_id": settings.agentkata_task_id,
                    "model": settings.agent_model,
                    "answer": result.output,
                    "submission": submit.data.model_dump(mode="json"),
                },
                indent=2,
            )
        )


if __name__ == "__main__":
    try:
        main()
    except AgentKataAPIError as exc:
        raise SystemExit(f"AgentKata API error: {exc.code}: {exc.message}") from exc
