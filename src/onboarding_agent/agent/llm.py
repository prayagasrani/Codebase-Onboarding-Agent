"""LLM access.

``GroqLLM`` speaks Groq's OpenAI-compatible chat-completions API over httpx,
translating the agent's provider-neutral tool schemas
(``name`` / ``description`` / ``input_schema``) to function-calling format.
It exposes ``complete`` (plain completion) and ``run_tools`` (manual tool-use
loop) and is built by ``build_llm`` from settings. When no ``GROQ_API_KEY`` is
configured the agent runs without an LLM and the graph composes extractive
answers from retrieved evidence instead — retrieval, tools, memory, and
evaluation all remain fully functional offline.
"""

from __future__ import annotations

import json
import re
import time
from collections.abc import Callable
from typing import TYPE_CHECKING

from onboarding_agent.logging import get_logger

if TYPE_CHECKING:
    from onboarding_agent.config import Settings

log = get_logger(__name__)

ToolExecuteFn = Callable[[str, dict], "ToolOutcome"]


class ToolOutcome:
    """Result of executing one tool call: payload + citation refs it grounds."""

    def __init__(self, payload: object, refs: list[str] | None = None, is_error: bool = False):
        self.payload = payload
        self.refs = refs or []
        self.is_error = is_error

    def as_text(self, limit: int = 6000) -> str:
        if isinstance(self.payload, str):
            return self.payload[:limit]
        return json.dumps(self.payload, default=str)[:limit]


GROQ_API = "https://api.groq.com/openai/v1"


class GroqToolCallError(RuntimeError):
    """The model generated a tool call that failed Groq's server-side schema validation."""


class GroqLLM:
    """Groq's OpenAI-compatible chat-completions API (plain completion + tool loop)."""

    def __init__(self, api_key: str, model: str = "openai/gpt-oss-120b",
                 max_tokens: int = 4096, client=None):
        import httpx

        self.model = model
        self.max_tokens = max_tokens
        self._client = client or httpx.Client(
            base_url=GROQ_API,
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=60.0,
        )

    def _chat(self, payload: dict, max_attempts: int = 4) -> dict:
        for attempt in range(max_attempts):
            resp = self._client.post("/chat/completions", json=payload)
            if resp.status_code == 429 and attempt < max_attempts - 1:
                # Free-tier TPM limits are per-minute; wait out the window and retry.
                wait = float(resp.headers.get("retry-after") or 0)
                if not wait:
                    m = re.search(r"try again in ([\d.]+)s", resp.text)
                    wait = float(m.group(1)) if m else 15.0
                wait = min(wait + 1.0, 90.0)
                log.warning("groq_rate_limited_retrying", wait_s=round(wait, 1))
                time.sleep(wait)
                continue
            if resp.status_code == 400 and "tool_use_failed" in resp.text:
                if attempt < max_attempts - 1:
                    # Low temperature makes the bad generation deterministic —
                    # bump it so the retry can actually produce a different call.
                    payload = {**payload,
                               "temperature": min(1.0, payload.get("temperature", 1.0) + 0.4)}
                    log.warning("groq_malformed_tool_call_retrying")
                    continue
                raise GroqToolCallError(resp.text[:400])
            if resp.status_code >= 400:
                raise RuntimeError(f"Groq API {resp.status_code}: {resp.text[:300]}")
            return resp.json()
        raise RuntimeError("Groq API: rate limit retries exhausted")

    # ------------------------------------------------------------------ plain completion
    def complete(self, prompt: str, system: str | None = None, max_tokens: int | None = None) -> str:
        messages: list[dict] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        data = self._chat({
            "model": self.model,
            "max_tokens": max_tokens or self.max_tokens,
            "messages": messages,
        })
        return data["choices"][0]["message"].get("content") or ""

    @staticmethod
    def _to_openai_tools(tools: list[dict]) -> list[dict]:
        """Agent tool schema (name/description/input_schema) -> OpenAI function-calling schema."""
        return [
            {
                "type": "function",
                "function": {
                    "name": t["name"],
                    "description": t.get("description", ""),
                    "parameters": t.get("input_schema", {"type": "object", "properties": {}}),
                },
            }
            for t in tools
        ]

    # ------------------------------------------------------------------ agentic tool loop
    def run_tools(
        self,
        system: str,
        user_message: str,
        tools: list[dict],
        execute: ToolExecuteFn,
        max_iterations: int = 8,
    ) -> tuple[str, list[dict]]:
        messages: list[dict] = [
            {"role": "system", "content": system},
            {"role": "user", "content": user_message},
        ]
        evidence: list[dict] = []
        oa_tools = self._to_openai_tools(tools)

        for _ in range(max_iterations):
            try:
                data = self._chat({
                    "model": self.model,
                    "max_tokens": self.max_tokens,
                    "messages": messages,
                    "tools": oa_tools,
                    "temperature": 0.2,  # open models mis-shape tool calls at high temp
                })
            except GroqToolCallError as exc:
                # Feed the validation error back so the model can self-correct.
                messages.append({
                    "role": "user",
                    "content": (
                        f"Your last tool call was rejected by schema validation: {exc}. "
                        "Re-issue it following the tool JSON schemas exactly — include "
                        "every required property, and pick the tool whose schema matches "
                        "your intent (e.g. use get_file with 'path' to read a file; "
                        "search_code requires a 'query' string)."
                    ),
                })
                continue
            msg = data["choices"][0]["message"]
            tool_calls = msg.get("tool_calls") or []
            if not tool_calls:
                return msg.get("content") or "", evidence

            messages.append({
                "role": "assistant",
                "content": msg.get("content"),
                "tool_calls": tool_calls,
            })
            for tc in tool_calls:
                name = tc["function"]["name"]
                try:
                    args = json.loads(tc["function"].get("arguments") or "{}")
                except json.JSONDecodeError:
                    args = {}
                outcome = execute(name, args)
                evidence.append({
                    "tool": name,
                    "input": args,
                    "refs": outcome.refs,
                    "result_snippet": outcome.as_text(1200),
                })
                # Keep tool results compact: free-tier Groq TPM budgets are small.
                content = outcome.as_text(2500)
                if outcome.is_error:
                    content = f"ERROR: {content}"
                messages.append({"role": "tool", "tool_call_id": tc["id"], "content": content})

        log.warning("tool_loop_max_iterations_reached")
        return "", evidence


LLM = GroqLLM


def build_llm(settings: Settings) -> GroqLLM | None:
    if not settings.groq_api_key:
        log.info("llm_disabled_offline_mode")
        return None
    log.info("llm_provider_selected", provider="groq", model=settings.groq_model)
    return GroqLLM(settings.groq_api_key, settings.groq_model, settings.llm_max_tokens)
