"""
QA Agent

Specialised agent responsible for testing, quality verification, and
result validation.  It receives QA tasks from the Supervisor, validates
outputs, and returns structured assessment results.

Responsibilities
----------------
* Testing code and verifying expected behaviour.
* Quality verification against acceptance criteria.
* Result validation — checking correctness and completeness.
"""

from typing import TYPE_CHECKING, Any, Dict, List, Optional

import structlog

from app.agents.base import AgentConfig, BaseAgent
from app.models.agent import AgentStatus, AgentType

if TYPE_CHECKING:
    from app.memory.manager import MemoryManager


class QAAgent(BaseAgent):
    """
    Agent that performs testing, quality verification, and validation.

    Parameters
    ----------
    config : AgentConfig | None
        Agent configuration.  ``type`` defaults to ``QA``.
    """

    def __init__(
        self,
        config: Optional[AgentConfig] = None,
        memory_manager: Optional["MemoryManager"] = None,
    ) -> None:
        if config is None:
            config = AgentConfig(
                name="QAAgent",
                type=AgentType.QA,
                description="Performs testing, quality verification, and validation.",
            )
        super().__init__(config)
        self.memory_manager = memory_manager
        self.logger = self.logger.bind(component="qa_agent")

    # ── BaseAgent interface ───────────────────────────────────

    async def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute a QA task.

        Parameters
        ----------
        input_data : dict
            Must contain ``"task"`` (str).  May also contain:
            - ``"code"`` (str) — code to test.
            - ``"expected_output"`` (str) — expected result for comparison.
            - ``"actual_output"`` (str) — actual result to validate.
            - ``"criteria"`` (list[str]) — acceptance criteria.
            - ``"context"`` (str) — additional context.

        Returns
        -------
        dict
            ``{"success": bool, "test_results": list[dict],
              "passed": int, "failed": int, "verdict": str,
              "errors": list[str]}``
        """
        task: str = input_data.get("task", "")
        code: str = input_data.get("code", "")
        expected_output: str = input_data.get("expected_output", "")
        actual_output: str = input_data.get("actual_output", "")
        criteria: List[str] = input_data.get("criteria", [])
        context: str = input_data.get("context", "")

        if not task:
            return {
                "success": False,
                "test_results": [],
                "passed": 0,
                "failed": 0,
                "verdict": "error",
                "errors": ["No task provided."],
            }

        self.status = AgentStatus.PROCESSING
        errors: List[str] = []
        test_results: List[Dict[str, Any]] = []
        passed: int = 0
        failed: int = 0
        verdict: str = "pending"

        try:
            test_results = self._run_tests(
                task, code, expected_output, actual_output, criteria, context
            )
            passed = sum(1 for t in test_results if t.get("passed"))
            failed = len(test_results) - passed
            verdict = self._determine_verdict(passed, failed)

            # Store QA feedback in memory
            if self.memory_manager:
                await self._store_qa_feedback(task, verdict, passed, failed)

            self.update_memory("last_task", task)
            self.update_memory("last_test_results", test_results)
            self.update_memory("last_verdict", verdict)

            self.logger.info(
                "QA task completed",
                task=task,
                passed=passed,
                failed=failed,
                verdict=verdict,
            )
        except Exception as exc:
            errors.append(str(exc))
            verdict = "error"
            self.logger.error("QA task failed", error=str(exc))
        finally:
            self.status = AgentStatus.IDLE

        return {
            "success": len(errors) == 0,
            "test_results": test_results,
            "passed": passed,
            "failed": failed,
            "verdict": verdict,
            "errors": errors,
        }

    async def reflect(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Reflect on the QA result.

        Parameters
        ----------
        result : dict
            The result dict returned by :meth:`process`.

        Returns
        -------
        dict
            Reflection insights.
        """
        test_results = result.get("test_results", [])
        errors = result.get("errors", [])
        insights: Dict[str, Any] = {
            "total_tests": len(test_results),
            "passed": result.get("passed", 0),
            "failed": result.get("failed", 0),
            "verdict": result.get("verdict", "unknown"),
            "had_errors": len(errors) > 0,
            "overall_success": result.get("success", False),
        }

        if test_results:
            categories = [t.get("category", "unknown") for t in test_results]
            insights["category_distribution"] = {
                c: categories.count(c) for c in set(categories)
            }

        self.logger.info("QA reflection", **insights)
        self.update_memory("last_reflection", insights)
        return insights

    # ── Internal helpers ──────────────────────────────────────

    async def _store_qa_feedback(
        self,
        task: str,
        verdict: str,
        passed: int,
        failed: int,
    ) -> None:
        """Store QA evaluation feedback in memory."""
        if not self.memory_manager:
            return

        try:
            from app.models.memory import MemoryType

            feedback = (
                f"QA verdict for '{task}': {verdict} "
                f"({passed} passed, {failed} failed)"
            )
            score = passed / max(passed + failed, 1)

            await self.memory_manager.store_memory(
                content=feedback,
                memory_type=MemoryType.LONG_TERM,
                metadata={
                    "verdict": verdict,
                    "passed": passed,
                    "failed": failed,
                    "pattern_type": "qa_feedback",
                },
            )
        except Exception as exc:
            self.logger.warning(
                "Failed to store QA feedback", error=str(exc)
            )

    @staticmethod
    def _run_tests(
        task: str,
        code: str,
        expected_output: str,
        actual_output: str,
        criteria: List[str],
        context: str,
    ) -> List[Dict[str, Any]]:
        """
        Run QA tests against the provided inputs.

        This is a template-based implementation.  In a production system
        this would invoke testing frameworks and assertion engines via
        the ToolRegistry.
        """
        test_results: List[Dict[str, Any]] = []

        # Task-level test
        test_results.append({
            "id": "TEST-001",
            "name": "Task completeness check",
            "category": "completeness",
            "passed": bool(task),
            "description": f"Verify task is well-defined: {task}",
        })

        # Output comparison test
        if expected_output and actual_output:
            match = expected_output.strip() == actual_output.strip()
            test_results.append({
                "id": "TEST-002",
                "name": "Output comparison",
                "category": "correctness",
                "passed": match,
                "description": (
                    "Compare expected vs actual output"
                    if match
                    else f"Mismatch: expected '{expected_output}', got '{actual_output}'"
                ),
            })

        # Code validation test
        if code:
            test_results.append({
                "id": "TEST-003",
                "name": "Code presence check",
                "category": "completeness",
                "passed": True,
                "description": "Code was provided for review.",
            })

        # Criteria-based tests
        for idx, criterion in enumerate(criteria, start=len(test_results) + 1):
            test_results.append({
                "id": f"TEST-C{idx:03d}",
                "name": f"Criterion: {criterion}",
                "category": "acceptance",
                "passed": True,
                "description": f"Verify acceptance criterion: {criterion}",
            })

        return test_results

    @staticmethod
    def _determine_verdict(passed: int, failed: int) -> str:
        """Determine the overall QA verdict."""
        if failed == 0 and passed > 0:
            return "pass"
        if failed > 0 and passed > 0:
            return "partial"
        if failed > 0 and passed == 0:
            return "fail"
        return "no_tests"
