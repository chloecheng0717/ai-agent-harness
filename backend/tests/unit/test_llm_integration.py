"""
Unit Tests for Orchestrator LLM Integration

Tests cover:
- Orchestrator with mocked LLM provider
- Lifecycle steps using LLM (understand_task, create_plan, evaluate_result, reflect, respond)
- Fallback behavior when LLM calls fail
- Token usage tracking through the LLM
- Full lifecycle run with LLM provider
"""

import json
import pytest
from typing import Any, Dict
from unittest.mock import AsyncMock, patch
from uuid import uuid4

from app.agents.core.orchestrator import Orchestrator
from app.models.llm import (
    ChatCompletionChoice,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatMessage,
    LLMProviderConfig,
    LLMProviderError,
    TokenUsageInfo,
)
from app.models.providers.openai_provider import OpenAIProvider
from app.models.state import (
    AgentState,
    LifecycleStep,
    PlanStep,
    StepStatus,
)


# ── Helpers ──────────────────────────────────────────────────────


def _make_llm_response(content: str, model: str = "test-model") -> ChatCompletionResponse:
    """Build a canned LLM response."""
    return ChatCompletionResponse(
        id="cmpl-test",
        model=model,
        choices=[
            ChatCompletionChoice(
                message=ChatMessage(role="assistant", content=content),
                finish_reason="stop",
            )
        ],
        usage=TokenUsageInfo(prompt_tokens=10, completion_tokens=20, total_tokens=30),
    )


def _make_provider() -> OpenAIProvider:
    """Build a provider with test credentials."""
    return OpenAIProvider(
        LLMProviderConfig(
            api_key="sk-test",
            base_url="https://api.example.com/v1",
            model="test-model",
        )
    )


# ── Orchestrator init with provider ──────────────────────────────


class TestOrchestratorWithProvider:
    def test_creates_with_provider(self) -> None:
        provider = _make_provider()
        orch = Orchestrator(llm_provider=provider)
        assert orch.llm_provider is provider

    def test_creates_without_provider(self) -> None:
        orch = Orchestrator()
        assert orch.llm_provider is None


# ── LLM helper ───────────────────────────────────────────────────


class TestLLMHelper:
    @pytest.mark.asyncio
    async def test_llm_chat_no_provider(self) -> None:
        orch = Orchestrator()
        state = AgentState()
        result = await orch._llm_chat(
            [ChatMessage(role="user", content="hi")],
            state,
        )
        assert result == ""

    @pytest.mark.asyncio
    async def test_llm_chat_with_provider(self) -> None:
        provider = _make_provider()
        orch = Orchestrator(llm_provider=provider)

        with patch.object(provider, "chat_completion", new_callable=AsyncMock) as mock:
            mock.return_value = _make_llm_response("Hello from LLM!")
            state = AgentState()
            result = await orch._llm_chat(
                [ChatMessage(role="user", content="hi")],
                state,
            )
            assert result == "Hello from LLM!"
            assert state.token_usage.total_tokens == 30
            assert state.token_usage.model == "test-model"
            assert state.token_usage.requests == 1

    @pytest.mark.asyncio
    async def test_llm_chat_error_recorded(self) -> None:
        provider = _make_provider()
        orch = Orchestrator(llm_provider=provider)

        with patch.object(provider, "chat_completion", new_callable=AsyncMock) as mock:
            mock.side_effect = LLMProviderError("server error")
            state = AgentState()
            with pytest.raises(LLMProviderError):
                await orch._llm_chat(
                    [ChatMessage(role="user", content="hi")],
                    state,
                )
            assert len(state.errors) == 1
            assert "server error" in state.errors[0]


# ── understand_task with LLM ─────────────────────────────────────


class TestUnderstandTaskLLM:
    @pytest.mark.asyncio
    async def test_uses_llm_to_extract_goal(self) -> None:
        provider = _make_provider()
        orch = Orchestrator(llm_provider=provider)

        with patch.object(provider, "chat_completion", new_callable=AsyncMock) as mock:
            mock.return_value = _make_llm_response("Scan the target network")
            state = AgentState()
            state.add_message("user", "please scan the target network for vulnerabilities")
            state = await orch.understand_task(state)
            assert state.goal == "Scan the target network"
            mock.assert_called_once()

    @pytest.mark.asyncio
    async def test_falls_back_on_llm_error(self) -> None:
        provider = _make_provider()
        orch = Orchestrator(llm_provider=provider)

        with patch.object(provider, "chat_completion", new_callable=AsyncMock) as mock:
            mock.side_effect = LLMProviderError("error")
            state = AgentState()
            state.add_message("user", "scan the network")
            state = await orch.understand_task(state)
            assert state.goal == "scan the network"

    @pytest.mark.asyncio
    async def test_uses_raw_message_without_provider(self) -> None:
        orch = Orchestrator()
        state = AgentState()
        state.add_message("user", "scan the network")
        state = await orch.understand_task(state)
        assert state.goal == "scan the network"


# ── create_plan with LLM ─────────────────────────────────────────


class TestCreatePlanLLM:
    @pytest.mark.asyncio
    async def test_uses_llm_to_create_plan(self) -> None:
        provider = _make_provider()
        orch = Orchestrator(llm_provider=provider)
        plan_json = json.dumps(
            [
                {"index": 0, "description": "Run nmap", "tool_name": "nmap", "rationale": "Scan ports"},
                {"index": 1, "description": "Analyse results", "tool_name": None, "rationale": "Summarise"},
            ]
        )

        with patch.object(provider, "chat_completion", new_callable=AsyncMock) as mock:
            mock.return_value = _make_llm_response(plan_json)
            state = AgentState()
            state.goal = "scan the network"
            state = await orch.create_plan(state)
            assert len(state.plan) == 2
            assert state.plan[0].tool_name == "nmap"
            assert state.plan[1].tool_name is None

    @pytest.mark.asyncio
    async def test_preserves_existing_plan(self) -> None:
        provider = _make_provider()
        orch = Orchestrator(llm_provider=provider)
        state = AgentState()
        existing = [PlanStep(index=0, description="custom")]
        state.plan = existing
        state = await orch.create_plan(state)
        assert state.plan == existing

    @pytest.mark.asyncio
    async def test_falls_back_on_invalid_json(self) -> None:
        provider = _make_provider()
        orch = Orchestrator(llm_provider=provider)

        with patch.object(provider, "chat_completion", new_callable=AsyncMock) as mock:
            mock.return_value = _make_llm_response("not valid json")
            state = AgentState()
            state.goal = "test"
            state = await orch.create_plan(state)
            assert len(state.plan) == 1
            assert "fallback" in state.plan[0].rationale.lower()

    @pytest.mark.asyncio
    async def test_falls_back_without_provider(self) -> None:
        orch = Orchestrator()
        state = AgentState()
        state.goal = "test"
        state = await orch.create_plan(state)
        assert len(state.plan) == 1


# ── evaluate_result with LLM ─────────────────────────────────────


class TestEvaluateResultLLM:
    @pytest.mark.asyncio
    async def test_uses_llm_to_evaluate(self) -> None:
        provider = _make_provider()
        orch = Orchestrator(llm_provider=provider)

        with patch.object(provider, "chat_completion", new_callable=AsyncMock) as mock:
            mock.return_value = _make_llm_response("Task completed successfully. All scans passed.")
            state = AgentState()
            state.goal = "scan network"
            state.results = [{"step_index": 0, "output": "scan done"}]
            state = await orch.evaluate_result(state)
            assert len(state.messages) == 1
            assert "successfully" in state.messages[0].content.lower()

    @pytest.mark.asyncio
    async def test_falls_back_on_llm_error(self) -> None:
        provider = _make_provider()
        orch = Orchestrator(llm_provider=provider)

        with patch.object(provider, "chat_completion", new_callable=AsyncMock) as mock:
            mock.side_effect = LLMProviderError("error")
            state = AgentState()
            state = await orch.evaluate_result(state)
            assert len(state.messages) == 1
            assert state.messages[0].role == "assistant"
            assert "execution completed" in state.messages[0].content.lower()

    @pytest.mark.asyncio
    async def test_heuristic_with_errors(self) -> None:
        orch = Orchestrator()
        state = AgentState()
        state.add_error("fail")
        state = await orch.evaluate_result(state)
        assert "error" in state.messages[0].content


# ── reflect with LLM ─────────────────────────────────────────────


class TestReflectLLM:
    @pytest.mark.asyncio
    async def test_uses_llm_to_reflect(self) -> None:
        provider = _make_provider()
        orch = Orchestrator(llm_provider=provider)

        with patch.object(provider, "chat_completion", new_callable=AsyncMock) as mock:
            mock.return_value = _make_llm_response("Good performance, but tool selection could improve.")
            state = AgentState()
            state.goal = "test"
            state.results = [{"output": "done"}]
            state = await orch.reflect(state)
            assert "reflection" in state.metadata
            assert "improve" in state.metadata["reflection"]

    @pytest.mark.asyncio
    async def test_falls_back_on_llm_error(self) -> None:
        provider = _make_provider()
        orch = Orchestrator(llm_provider=provider)

        with patch.object(provider, "chat_completion", new_callable=AsyncMock) as mock:
            mock.side_effect = LLMProviderError("error")
            state = AgentState()
            state = await orch.reflect(state)
            assert "reflection" not in state.metadata

    @pytest.mark.asyncio
    async def test_logs_summary_without_provider(self) -> None:
        orch = Orchestrator()
        state = AgentState()
        state = await orch.reflect(state)
        assert state.goal == ""


# ── respond with LLM ─────────────────────────────────────────────


class TestRespondLLM:
    @pytest.mark.asyncio
    async def test_uses_llm_to_respond(self) -> None:
        provider = _make_provider()
        orch = Orchestrator(llm_provider=provider)

        with patch.object(provider, "chat_completion", new_callable=AsyncMock) as mock:
            mock.return_value = _make_llm_response("The network scan found 3 open ports.")
            state = AgentState()
            state.goal = "scan network"
            state.results = [{"output": "port 22,80,443 open"}]
            state = await orch.respond(state)
            assert state.final_answer == "The network scan found 3 open ports."
            assert state.status == StepStatus.COMPLETED

    @pytest.mark.asyncio
    async def test_falls_back_on_llm_error(self) -> None:
        provider = _make_provider()
        orch = Orchestrator(llm_provider=provider)

        with patch.object(provider, "chat_completion", new_callable=AsyncMock) as mock:
            mock.side_effect = LLMProviderError("error")
            state = AgentState()
            state.results = [{"output": "result"}]
            state = await orch.respond(state)
            assert "result" in state.final_answer

    @pytest.mark.asyncio
    async def test_uses_fallback_answer_when_llm_empty(self) -> None:
        provider = _make_provider()
        orch = Orchestrator(llm_provider=provider)

        with patch.object(provider, "chat_completion", new_callable=AsyncMock) as mock:
            mock.return_value = _make_llm_response("")
            state = AgentState()
            state.results = [{"output": "result"}]
            state = await orch.respond(state)
            assert "result" in state.final_answer

    @pytest.mark.asyncio
    async def test_fallback_answer_no_results(self) -> None:
        orch = Orchestrator()
        state = AgentState()
        state = await orch.respond(state)
        assert state.final_answer == "No result produced."


# ── Full lifecycle with LLM ──────────────────────────────────────


class TestFullLifecycleWithLLM:
    @pytest.mark.asyncio
    async def test_full_lifecycle_with_provider(self) -> None:
        provider = _make_provider()
        orch = Orchestrator(llm_provider=provider)

        plan_json = json.dumps(
            [{"index": 0, "description": "scan", "tool_name": None, "rationale": "scan"}]
        )

        responses = [
            _make_llm_response("Scan the target"),        # understand_task
            _make_llm_response(plan_json),                 # create_plan
            _make_llm_response("COMPLETE: YES - scan complete"),  # evaluate_result
            _make_llm_response("All steps completed well"),# reflect
            _make_llm_response("Scan complete, 3 ports open"),  # respond
        ]

        call_idx = 0

        async def mock_chat(request: ChatCompletionRequest) -> ChatCompletionResponse:
            nonlocal call_idx
            resp = responses[call_idx]
            call_idx += 1
            return resp

        with patch.object(provider, "chat_completion", side_effect=mock_chat):
            result = await orch.run(
                task_id=uuid4(),
                task_data={"description": "scan target 192.168.1.1"},
            )

        assert result["status"] == "completed"
        assert result["final_answer"] == "Scan complete, 3 ports open"
        assert result["token_usage"]["total_tokens"] > 0
        assert result["token_usage"]["requests"] == 5

    @pytest.mark.asyncio
    async def test_full_lifecycle_without_provider(self) -> None:
        orch = Orchestrator()
        result = await orch.run(
            task_id=uuid4(),
            task_data={"description": "scan target"},
        )
        assert result["status"] == "completed"


# ── close() ──────────────────────────────────────────────────────


class TestOrchestratorClose:
    @pytest.mark.asyncio
    async def test_close_with_provider(self) -> None:
        provider = _make_provider()
        orch = Orchestrator(llm_provider=provider)
        with patch.object(provider, "close", new_callable=AsyncMock) as mock:
            await orch.close()
            mock.assert_called_once()

    @pytest.mark.asyncio
    async def test_close_without_provider(self) -> None:
        orch = Orchestrator()
        await orch.close()
