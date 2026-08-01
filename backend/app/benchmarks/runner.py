"""
CTF Benchmark Runner

Runs a :class:`Challenge` end to end through the real agent pipeline
(orchestrator → tool registry → sandbox) using only the existing CTF
tools, then produces a :class:`ChallengeResult` with pass/fail, the
failed milestone, tools used, execution time, and iteration count.
"""

from __future__ import annotations

import time
from typing import Any
from uuid import uuid4

import structlog

from app.benchmarks.models import (
    Challenge,
    ChallengeResult,
    RunTrace,
    evaluate_challenge,
)
from app.config import get_settings
from app.models.llm import BaseLLMProvider
from app.sandbox.docker_sandbox import DockerSandbox, SandboxConfig
from app.tools.plugins.archive_tool import ArchiveTool
from app.tools.plugins.encoding_tool import EncodingTool
from app.tools.plugins.file_tool import FileTool
from app.tools.plugins.python_tool import PythonTool
from app.tools.plugins.strings_tool import StringsTool
from app.tools.registry import ToolRegistry

#: The complete set of tools a challenge run may use.  No new tools here --
#: this is exactly the current CTF toolchain, surfaced for measurement.
_CTF_TOOL_CLASSES = (
    FileTool,
    StringsTool,
    PythonTool,
    EncodingTool,
    ArchiveTool,
)


class BenchmarkRunner:
    """
    Executes challenges against the current agent capability.

    Parameters
    ----------
    settings : Settings | None
        App settings; defaults to the cached singleton.
    sandbox : DockerSandbox | None
        A pre-built sandbox (used by tests with a simulated client).  When
        omitted the runner builds one from the settings.
    llm_provider : BaseLLMProvider | None
        Provider to drive the orchestrator.  When omitted the runner builds
        one fresh per challenge.
    verbose : bool
        Print LLM requests, plans, and tool calls during a run.
    max_iterations : int | None
        Cap the orchestrator's plan-execute-evaluate iterations.
    """

    def __init__(
        self,
        settings: Any = None,
        sandbox: DockerSandbox | None = None,
        llm_provider: BaseLLMProvider | None = None,
        verbose: bool = False,
        max_iterations: int | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.sandbox = sandbox
        self.llm_provider = llm_provider
        self.verbose = verbose
        self.max_iterations = max_iterations
        self.logger = structlog.get_logger().bind(component="benchmark_runner")

    # ── construction helpers ────────────────────────────────────

    def _build_sandbox(self) -> DockerSandbox:
        config = SandboxConfig(
            image=self.settings.sandbox_image,
            memory_limit=self.settings.sandbox_memory_limit,
            cpu_limit=self.settings.sandbox_cpu_limit,
            timeout=self.settings.sandbox_timeout,
            network_disabled=self.settings.sandbox_network_disabled,
        )
        return DockerSandbox(config)

    def _build_provider(self) -> BaseLLMProvider:
        # Imported lazily to avoid hard dependency on the HTTP provider for
        # unit tests (which inject a stub provider instead).
        from app.models.providers.openai_provider import OpenAIProvider

        return OpenAIProvider()

    @staticmethod
    def _build_registry(sandbox: DockerSandbox) -> ToolRegistry:
        registry = ToolRegistry(sandbox=sandbox)
        for tool_class in _CTF_TOOL_CLASSES:
            registry.register_instance(tool_class(sandbox))
        return registry

    # ── single challenge ─────────────────────────────────────────

    async def run_challenge(
        self,
        challenge: Challenge,
        llm_provider: BaseLLMProvider | None = None,
    ) -> ChallengeResult:
        """
        Run one challenge and evaluate the result.

        Parameters
        ----------
        challenge : Challenge
            The challenge to execute.
        llm_provider : BaseLLMProvider | None
            Provider override for this run (falls back to the runner's
            provider, then to a freshly built one).

        Returns
        -------
        ChallengeResult
            Full per-challenge outcome.
        """
        # Imported lazily so benchmark modules never touch orchestrator
        # imports unless a run is actually requested.
        from app.benchmarks.orchestrator import BenchmarkOrchestrator

        provider = llm_provider or self.llm_provider or self._build_provider()

        sandbox = self.sandbox or self._build_sandbox()
        registry = self._build_registry(sandbox)

        orchestrator = BenchmarkOrchestrator(
            llm_provider=provider,
            tool_registry=registry,
            sandbox=sandbox,
            challenge_files=challenge.files,
            verbose=self.verbose,
            max_iterations=self.max_iterations,
        )

        start = time.monotonic()
        try:
            result: dict[str, Any] = await orchestrator.run(
                task_id=uuid4(),
                task_data={"description": challenge.goal},
                tools_available=challenge.required_capabilities,
            )
        except Exception as exc:
            self.logger.error(
                "Challenge run crashed",
                challenge=challenge.name,
                error=str(exc),
            )
            result = {
                "status": "crashed",
                "final_answer": "",
                "errors": [str(exc)],
                "tools_used": [],
                "step_history": [],
                "iterations": 0,
            }
        duration_ms = (time.monotonic() - start) * 1000

        trace = RunTrace(
            tool_calls=result.get("benchmark_trace", []),
            workspace_files=result.get("benchmark_workspace_files", []),
            final_answer=result.get("final_answer") or "",
            errors=result.get("errors", []),
            status=result.get("status", ""),
            iterations=result.get("iterations", 0),
            execution_time_ms=duration_ms,
        )

        passed, flag_found, milestones, failed_step = evaluate_challenge(
            challenge, trace
        )

        return ChallengeResult(
            name=challenge.name,
            category=challenge.category,
            difficulty=challenge.difficulty,
            passed=passed,
            flag_found=flag_found,
            status=trace.status,
            failed_step=failed_step,
            milestones=milestones,
            tools_used=list(result.get("tools_used", [])),
            execution_time_ms=duration_ms,
            iterations=trace.iterations,
            errors=list(trace.errors),
            expected_flag=challenge.expected_flag,
            final_answer=trace.final_answer,
        )

    # ── multiple challenges ──────────────────────────────────────

    async def run_challenges(
        self,
        challenges: list[Challenge],
    ) -> list[ChallengeResult]:
        """
        Run a list of challenges sequentially, one fresh LLM provider per
        challenge so the runs are isolated from one another.
        """
        results: list[ChallengeResult] = []
        for challenge in challenges:
            self.logger.info(
                "Running challenge",
                challenge=challenge.name,
                difficulty=challenge.difficulty,
            )
            results.append(await self.run_challenge(challenge))
        return results
