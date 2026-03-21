import json
import os
from dataclasses import dataclass
from typing import TypedDict

from agentkata import AgentKataAPIError, Client, RequestMeta
from langchain_anthropic import ChatAnthropic
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

SEARCH_PAGE_LIMIT = 2
EXCEPTIONS_QUERY = "exceptions escalation review global policy"

DECISION_PROMPT = """\
You are a corporate policy analyst. Decide whether to allow, deny, or review.

Decision logic (follow in order):
1. Identify employee's country, employment_type, hire date, and role.
2. Find the ACTIVE policy document. Ignore archived documents entirely.
3. Check eligibility criteria precisely:
   - Dates: 2025-06-10 is AFTER 2025-01-01 (employee qualifies).
   - Amounts: USD 500 < USD 600 (within limit).
   - Employment type must match exactly.
4. ALLOW: active policy explicitly permits and all criteria are met.
5. DENY: active policy explicitly prohibits (wrong employment type,
   amount exceeds limit, item not an eligible expense category).
6. REVIEW: specific policy excludes this case (e.g. 'meals excluded')
   BUT an exceptions/escalation policy routes edge cases to HR review.
   If you see an exceptions policy, prefer 'review' over 'deny'.

Citation rules:
- Only cite doc_id values from ACTIVE documents. Never cite archived.
- For 'review', cite the exceptions policy, not the excluding one.
- Include at least one citation.

Request:
{request}

Employee:
{employee}

Requester:
{requester}

Manager:
{manager}

Documents:
{documents}"""


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    agentkata_base_url: str
    agentkata_api_token: str
    agentkata_track_id: str = "corporate-ai"
    agentkata_task_id: str = "policy-answer-with-citations"
    llm_provider: str = "anthropic"
    llm_model: str = "claude-haiku-4-5"
    openai_api_key: str = ""
    anthropic_api_key: str = ""


@dataclass(frozen=True)
class SolverDeps:
    client: Client
    track_id: str
    task_id: str


class PolicyAnswer(BaseModel):
    decision: str = Field(pattern="^(allow|deny|review)$")
    summary: str = Field(min_length=1)
    citations: list[str] = Field(min_length=1)


class GraphState(TypedDict, total=False):
    request: dict[str, object]
    employee: dict[str, object]
    requester: dict[str, object]
    manager: dict[str, object]
    search_offset: int
    search_has_more: bool
    search_items: list[dict[str, object]]
    documents: list[dict[str, object]]
    answer: dict[str, object]
    model_name: str
    prompt_tokens: int
    completion_tokens: int
    submission: dict[str, object]


def build_model(settings: Settings) -> BaseChatModel:
    provider = settings.llm_provider.lower()

    if provider == "openai":
        if not settings.openai_api_key:
            raise SystemExit("OPENAI_API_KEY required when LLM_PROVIDER=openai")
        return ChatOpenAI(model=settings.llm_model, temperature=0)

    if provider == "anthropic":
        if not settings.anthropic_api_key:
            raise SystemExit("ANTHROPIC_API_KEY required when LLM_PROVIDER=anthropic")
        return ChatAnthropic(model=settings.llm_model, temperature=0)

    raise SystemExit(f"LLM_PROVIDER must be 'openai' or 'anthropic', got '{provider}'")


def task_action(
    deps: SolverDeps,
    action: str,
    payload: dict[str, object] | None = None,
) -> dict[str, object]:
    return deps.client.track_task_action(
        track_id=deps.track_id,
        task_id=deps.task_id,
        action=action,
        payload=payload or {},
    ).data


def lookup_person(deps: SolverDeps, person_id: str) -> dict[str, object]:
    try:
        return task_action(deps, "people/get", {"person_id": person_id})
    except AgentKataAPIError as exc:
        if exc.status_code == 400:
            return {}
        raise


def search_page(
    deps: SolverDeps, query: str, *, limit: int, offset: int
) -> tuple[list[dict[str, object]], bool, int]:
    response = task_action(
        deps, "documents/search", {"query": query, "limit": limit, "offset": offset}
    )
    items = response["items"]
    pagination = response["pagination"]
    if not isinstance(items, list) or not isinstance(pagination, dict):
        raise SystemExit("documents/search returned unexpected shape")

    has_more = bool(pagination.get("has_more"))
    next_offset = pagination.get("next_offset")
    resolved_offset = int(next_offset) if isinstance(next_offset, int) else offset
    return [item for item in items if isinstance(item, dict)], has_more, resolved_offset


def build_graph(deps: SolverDeps, model: BaseChatModel):
    def bootstrap_context(_state: GraphState) -> GraphState:
        request = task_action(deps, "request")
        employee = task_action(
            deps, "hr/employees/get", {"employee_id": request["subject_employee_id"]}
        )

        requester_data = request["requester"]
        if not isinstance(requester_data, dict):
            raise SystemExit("requester payload is missing")
        requester = task_action(deps, "people/get", {"person_id": requester_data["person_id"]})

        manager_id = employee.get("manager_person_id")
        if manager_id == requester.get("person_id"):
            manager = requester
        elif isinstance(manager_id, str) and manager_id:
            manager = lookup_person(deps, manager_id)
        else:
            manager = {}

        return {
            "request": request,
            "employee": employee,
            "requester": requester,
            "manager": manager,
            "search_offset": 0,
            "search_has_more": True,
            "search_items": [],
        }

    def search_documents(state: GraphState) -> GraphState:
        request = state["request"]
        employee = state["employee"]
        query = (
            f"{employee['country']} {employee['employment_type']}"
            f" {employee['job_family']} {request['question']}"
        )

        items, has_more, next_offset = search_page(
            deps, query, limit=SEARCH_PAGE_LIMIT, offset=state["search_offset"]
        )
        return {
            "search_items": [*state["search_items"], *items],
            "search_has_more": has_more,
            "search_offset": next_offset,
        }

    def search_exceptions(state: GraphState) -> GraphState:
        items, _, _ = search_page(deps, EXCEPTIONS_QUERY, limit=5, offset=0)
        seen = {item.get("doc_id") for item in state["search_items"]}
        new_items = [item for item in items if item.get("doc_id") not in seen]
        return {"search_items": [*state["search_items"], *new_items]}

    def fetch_documents(state: GraphState) -> GraphState:
        documents = []
        for item in state["search_items"]:
            doc_id = item.get("doc_id")
            if isinstance(doc_id, str) and doc_id:
                documents.append(task_action(deps, "documents/get", {"doc_id": doc_id}))
        return {"documents": documents}

    def decide_answer(state: GraphState) -> GraphState:
        structured = model.with_structured_output(PolicyAnswer, method="json_schema")
        prompt = DECISION_PROMPT.format(
            request=json.dumps(state["request"], indent=2),
            employee=json.dumps(state["employee"], indent=2),
            requester=json.dumps(state["requester"], indent=2),
            manager=json.dumps(state["manager"], indent=2),
            documents=json.dumps(state["documents"], indent=2),
        )
        result = structured.invoke(prompt)
        return {
            "answer": result.model_dump(mode="json"),
            "prompt_tokens": 0,
            "completion_tokens": 0,
        }

    def submit_answer(state: GraphState) -> GraphState:
        submit = deps.client.submit_track_task(
            track_id=deps.track_id,
            task_id=deps.task_id,
            answer=state["answer"],
            meta=RequestMeta(
                model=state["model_name"],
                prompt_tokens=state["prompt_tokens"],
                completion_tokens=state["completion_tokens"],
            ),
        )
        return {"submission": submit.data.model_dump(mode="json")}

    def should_paginate(state: GraphState) -> str:
        return "continue" if state["search_has_more"] else "done"

    graph = StateGraph(GraphState)
    graph.add_node("bootstrap_context", bootstrap_context)
    graph.add_node("search_documents", search_documents)
    graph.add_node("search_exceptions", search_exceptions)
    graph.add_node("fetch_documents", fetch_documents)
    graph.add_node("decide_answer", decide_answer)
    graph.add_node("submit_answer", submit_answer)

    graph.add_edge(START, "bootstrap_context")
    graph.add_edge("bootstrap_context", "search_documents")
    graph.add_conditional_edges(
        "search_documents",
        should_paginate,
        {"continue": "search_documents", "done": "search_exceptions"},
    )
    graph.add_edge("search_exceptions", "fetch_documents")
    graph.add_edge("fetch_documents", "decide_answer")
    graph.add_edge("decide_answer", "submit_answer")
    graph.add_edge("submit_answer", END)
    return graph.compile()


def main() -> None:
    settings = Settings()
    os.environ.setdefault("OPENAI_API_KEY", settings.openai_api_key)
    os.environ.setdefault("ANTHROPIC_API_KEY", settings.anthropic_api_key)
    model = build_model(settings)

    with Client(
        base_url=settings.agentkata_base_url,
        api_token=settings.agentkata_api_token,
    ) as client:
        deps = SolverDeps(
            client=client,
            track_id=settings.agentkata_track_id,
            task_id=settings.agentkata_task_id,
        )
        client.restart_track(track_id=deps.track_id)
        result = build_graph(deps, model).invoke({"model_name": settings.llm_model})

    print(
        json.dumps(
            {
                "track_id": deps.track_id,
                "task_id": deps.task_id,
                "model": settings.llm_model,
                "answer": result["answer"],
                "submission": result["submission"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    try:
        main()
    except AgentKataAPIError as exc:
        raise SystemExit(f"AgentKata API error: {exc.code}: {exc.message}") from exc
