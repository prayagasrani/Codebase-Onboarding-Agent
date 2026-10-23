"""GroqLLM adapter: OpenAI-compatible completions + tool loop (httpx MockTransport)."""

from __future__ import annotations

import json

import httpx

from onboarding_agent.agent.llm import GroqLLM, ToolOutcome, build_llm
from onboarding_agent.config import Settings


def make_llm(handler) -> GroqLLM:
    client = httpx.Client(base_url="https://api.groq.com/openai/v1",
                          transport=httpx.MockTransport(handler))
    return GroqLLM(api_key="test-key", client=client)


def _completion(content: str | None = None, tool_calls: list | None = None) -> dict:
    msg: dict = {"role": "assistant", "content": content}
    if tool_calls:
        msg["tool_calls"] = tool_calls
    return {"choices": [{"message": msg, "finish_reason": "tool_calls" if tool_calls else "stop"}]}


def test_complete_returns_text():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/openai/v1/chat/completions"
        body = json.loads(request.content)
        assert body["messages"][0] == {"role": "system", "content": "be brief"}
        assert body["messages"][1]["role"] == "user"
        return httpx.Response(200, json=_completion("hello from groq"))

    assert make_llm(handler).complete("hi", system="be brief") == "hello from groq"


def test_run_tools_executes_and_returns_final_answer():
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        body = json.loads(request.content)
        if calls["n"] == 1:
            assert body["tools"][0]["type"] == "function"
            assert body["tools"][0]["function"]["name"] == "search_code"
            return httpx.Response(200, json=_completion(tool_calls=[{
                "id": "call_1", "type": "function",
                "function": {"name": "search_code",
                             "arguments": json.dumps({"query": "auth"})},
            }]))
        # Second round: the tool result must be threaded back in.
        roles = [m["role"] for m in body["messages"]]
        assert roles == ["system", "user", "assistant", "tool"]
        assert body["messages"][3]["tool_call_id"] == "call_1"
        return httpx.Response(200, json=_completion("answer [auth/service.py:1-10]"))

    def execute(name: str, args: dict) -> ToolOutcome:
        assert name == "search_code" and args == {"query": "auth"}
        return ToolOutcome({"hits": 1}, refs=["auth/service.py:1-10"])

    tools = [{"name": "search_code", "description": "search",
              "input_schema": {"type": "object", "properties": {"query": {"type": "string"}}}}]
    final, evidence = make_llm(handler).run_tools("sys", "how does auth work?", tools, execute)
    assert final == "answer [auth/service.py:1-10]"
    assert evidence[0]["tool"] == "search_code"
    assert evidence[0]["refs"] == ["auth/service.py:1-10"]


def test_run_tools_error_result_is_flagged():
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] == 1:
            return httpx.Response(200, json=_completion(tool_calls=[{
                "id": "call_1", "type": "function",
                "function": {"name": "get_file", "arguments": "{}"},
            }]))
        body = json.loads(request.content)
        assert body["messages"][3]["content"].startswith("ERROR:")
        return httpx.Response(200, json=_completion("done"))

    def execute(name: str, args: dict) -> ToolOutcome:
        return ToolOutcome("file not found", is_error=True)

    tools = [{"name": "get_file", "description": "", "input_schema": {"type": "object"}}]
    final, _ = make_llm(handler).run_tools("sys", "q", tools, execute)
    assert final == "done"


def test_run_tools_recovers_from_tool_validation_400():
    calls = {"n": 0}
    reject = {"error": {"message": "Tool call validation failed: missing properties: 'query'",
                        "code": "tool_use_failed"}}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] <= 4:  # exhaust _chat's internal retries -> GroqToolCallError
            return httpx.Response(400, json=reject)
        body = json.loads(request.content)
        # The corrective feedback message must have been appended.
        assert "rejected by schema validation" in body["messages"][-1]["content"]
        return httpx.Response(200, json=_completion("recovered"))

    tools = [{"name": "search_code", "description": "",
              "input_schema": {"type": "object", "required": ["query"]}}]
    final, _ = make_llm(handler).run_tools("sys", "q", tools, lambda n, a: ToolOutcome("x"))
    assert final == "recovered"


def _settings(**overrides) -> Settings:
    base = {"groq_api_key": "", "github_token": "", "github_repo": ""}
    return Settings(**{**base, **overrides})


def test_build_llm_provider_selection():
    # No key -> offline/extractive mode.
    assert build_llm(_settings()) is None
    # Key present -> Groq, using the configured model.
    llm = build_llm(_settings(groq_api_key="g", groq_model="llama-3.3-70b-versatile"))
    assert isinstance(llm, GroqLLM)
    assert llm.model == "llama-3.3-70b-versatile"
