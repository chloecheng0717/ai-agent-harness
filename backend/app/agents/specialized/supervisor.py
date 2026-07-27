"""
Supervisor Agent

The SupervisorAgent is the top-level coordinator for multi-agent workflows.
It receives user requests, analyses task requirements, decides which
specialised agents are needed, assigns tasks, coordinates execution,
collects responses, and generates the final combined result.

Responsibilities
----------------
* Receive and analyse user requests.
* Determine which specialised agents to invoke.
* Dispatch tasks via :class:`AgentMessage` objects.
* Coordinate sequential or parallel agent execution.
* Collect and synthesise results into a unified response.
"""

from typing import TYPE_CHECKING, Any, Dict, List, Optional

import structlog

from app.agents.base import AgentConfig, BaseAgent
from app.agents.specialized.coding import CodingAgent
from app.agents.specialized.qa import QAAgent
from app.agents.specialized.research import ResearchAgent
from app.agents.specialized.security import SecurityAgent
from app.communication.agent_message import AgentMessage, MessageStatus
from app.models.agent import AgentStatus, AgentType
from app.models.memory import EpisodicMemoryEntry, MemoryType

if TYPE_CHECKING:
    from app.memory.manager import MemoryManager
    from app.rag.pipeline import RAGPipeline


class SupervisorAgent(BaseAgent):
    """
    Top-level coordinator that dispatches tasks to specialised agents.

    Parameters
    ----------
    config : AgentConfig | None
        Agent configuration.  ``type`` defaults to ``SUPERVISOR``.
    registry : AgentRegistry | None
        Optional pre-configured agent registry.  When ``None`` a default
        registry is created with the four built-in specialised agents.
    """

    def __init__(
        self,
        config: Optional[AgentConfig] = None,
        registry: Optional["AgentRegistry"] = None,
        memory_manager: Optional["MemoryManager"] = None,
        rag_pipeline: Optional["RAGPipeline"] = None,
    ) -> None:
        if config is None:
            config = AgentConfig(
                name="SupervisorAgent",
                type=AgentType.SUPERVISOR,
                description="Coordinates specialised agents to fulfil user requests.",
            )
        super().__init__(config)
        self.logger = self.logger.bind(component="supervisor_agent")
        self.memory_manager = memory_manager
        self.rag_pipeline = rag_pipeline

        # Lazy import to avoid circular dependency at module level
        if registry is None:
            from app.agents.specialized.registry import AgentRegistry
            self._registry = AgentRegistry()
            self._register_defaults()
        else:
            self._registry = registry

        self._message_log: List[AgentMessage] = []

    # ── Default agent registration ────────────────────────────

    def _register_defaults(self) -> None:
        """Register the four built-in specialised agents."""
        self._registry.register("researcher", ResearchAgent)
        self._registry.register("coder", CodingAgent)
        self._registry.register("security", SecurityAgent)
        self._registry.register("qa", QAAgent)
        self.logger.debug("Default agents registered")

    # ── BaseAgent interface ───────────────────────────────────

    async def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Coordinate a multi-agent workflow for the given request.

        Parameters
        ----------
        input_data : dict
            Must contain ``"task"`` (str).  May also contain:
            - ``"context"`` (str) — additional context.
            - ``"agents"`` (list[str]) — specific agent keys to use.

        Returns
        -------
        dict
            ``{"success": bool, "final_result": str,
              "agent_results": dict, "messages": list[dict],
              "errors": list[str]}``
        """
        task: str = input_data.get("task", "")
        context: str = input_data.get("context", "")
        agent_keys: Optional[List[str]] = input_data.get("agents")

        if not task:
            return {
                "success": False,
                "final_result": "",
                "agent_results": {},
                "messages": [],
                "errors": ["No task provided."],
            }

        self.status = AgentStatus.PROCESSING
        errors: List[str] = []
        agent_results: Dict[str, Any] = {}
        messages: List[AgentMessage] = []

        try:
            # 0. Retrieve relevant past tasks from memory
            past_context = ""
            if self.memory_manager:
                past_context = await self._retrieve_past_context(task, agent_keys)

            # 0b. Retrieve relevant knowledge base context
            kb_context = ""
            if self.rag_pipeline:
                kb_context = await self._retrieve_knowledge_context(task)

            combined_context = "\n\n".join(
                filter(None, [context, past_context, kb_context])
            )

            # 1. Analyse and decide which agents are needed
            selected_agents = self._select_agents(task, agent_keys)
            self.logger.info(
                "Agents selected",
                task=task,
                agents=selected_agents,
            )

            # 2. Dispatch tasks and collect responses
            for agent_key in selected_agents:
                agent_cls = self._registry.get(agent_key)
                if agent_cls is None:
                    errors.append(f"Agent '{agent_key}' not found in registry")
                    continue

                msg = self._create_message(
                    sender=self.config.name,
                    receiver=agent_key,
                    task=task,
                    context=context,
                )
                messages.append(msg)

                result = await self._dispatch_to_agent(
                    agent_key, agent_cls, task, combined_context,
                )
                agent_results[agent_key] = result

                if result.get("success"):
                    msg.mark_completed(str(result))
                else:
                    err = "; ".join(result.get("errors", ["Unknown error"]))
                    msg.mark_failed(err)
                    errors.append(f"Agent '{agent_key}' failed: {err}")

            # 3. Synthesise final result
            final_result = self._synthesise(task, agent_results)

            # 4. Store completed workflow in memory
            if self.memory_manager:
                await self._store_workflow(
                    task, selected_agents, agent_results, final_result, errors
                )

            self.update_memory("last_task", task)
            self.update_memory("last_agent_results", agent_results)
            self.update_memory("last_final_result", final_result)

            self.logger.info(
                "Supervision completed",
                task=task,
                agents_dispatched=len(selected_agents),
                errors=len(errors),
            )
        except Exception as exc:
            errors.append(str(exc))
            self.logger.error("Supervision failed", error=str(exc))
        finally:
            self.status = AgentStatus.IDLE

        self._message_log.extend(messages)

        return {
            "success": len(errors) == 0,
            "final_result": final_result if not errors else "",
            "agent_results": agent_results,
            "messages": [m.to_dict() for m in messages],
            "errors": errors,
        }

    async def reflect(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Reflect on the supervision result.

        Parameters
        ----------
        result : dict
            The result dict returned by :meth:`process`.

        Returns
        -------
        dict
            Reflection insights.
        """
        agent_results = result.get("agent_results", {})
        errors = result.get("errors", [])
        insights: Dict[str, Any] = {
            "agents_dispatched": len(agent_results),
            "had_errors": len(errors) > 0,
            "overall_success": result.get("success", False),
            "final_result_length": len(result.get("final_result", "")),
        }

        successful = sum(
            1 for r in agent_results.values() if r.get("success")
        )
        insights["successful_agents"] = successful
        insights["failed_agents"] = len(agent_results) - successful

        self.logger.info("Supervisor reflection", **insights)
        self.update_memory("last_reflection", insights)
        return insights

    # ── Agent selection ───────────────────────────────────────

    def _select_agents(
        self,
        task: str,
        explicit_agents: Optional[List[str]] = None,
    ) -> List[str]:
        """
        Decide which agents should handle the task.

        If the caller provides explicit agent keys those are used.
        Otherwise a keyword-based heuristic maps task content to agents.
        """
        if explicit_agents:
            return [a for a in explicit_agents if self._registry.exists(a)]

        task_lower = task.lower()
        selected: List[str] = []

        keyword_map = {
            "researcher": [
                "research", "gather", "find", "search", "summarise",
                "summarize", "information", "knowledge", "analyse", "analyze",
            ],
            "coder": [
                "code", "write", "implement", "debug", "fix", "program",
                "function", "class", "api", "script",
            ],
            "security": [
                "security", "vulnerability", "vulnerabilities", "scan",
                "risk", "threat", "exploit", "cve", "audit",
            ],
            "qa": [
                "test", "verify", "validate", "check", "quality",
                "assertion", "assert", "coverage", "pass", "fail",
            ],
        }

        for agent_key, keywords in keyword_map.items():
            if any(kw in task_lower for kw in keywords):
                selected.append(agent_key)

        # Always include researcher as a baseline if nothing matched
        if not selected:
            selected.append("researcher")

        return selected

    # ── Message creation ──────────────────────────────────────

    def _create_message(
        self,
        sender: str,
        receiver: str,
        task: str,
        context: str = "",
    ) -> AgentMessage:
        """Create an AgentMessage for dispatching a task."""
        full_task = task
        if context:
            full_task = f"{task}\n\nContext: {context}"

        msg = AgentMessage(
            sender=sender,
            receiver=receiver,
            task=full_task,
        )
        msg.mark_sent()
        self.logger.debug(
            "Message created",
            sender=sender,
            receiver=receiver,
        )
        return msg

    # ── Dispatch ──────────────────────────────────────────────

    @staticmethod
    async def _dispatch_to_agent(
        agent_key: str,
        agent_cls: type,
        task: str,
        context: str,
    ) -> Dict[str, Any]:
        """
        Instantiate (or reuse) an agent and process the task.

        For simplicity, a new instance is created for each dispatch.
        In a production system the registry could maintain singletons.
        """
        agent = agent_cls()
        input_data: Dict[str, Any] = {"task": task}
        if context:
            input_data["context"] = context
        return await agent.process(input_data)

    # ── Synthesis ─────────────────────────────────────────────

    @staticmethod
    def _synthesise(
        task: str,
        agent_results: Dict[str, Any],
    ) -> str:
        """
        Synthesise a final combined result from all agent outputs.

        This is a template-based implementation.  In a production system
        this would invoke an LLM to produce a coherent narrative.
        """
        if not agent_results:
            return f"No agent results available for task: {task}"

        parts: List[str] = [f"Supervisor Result for: {task}", ""]

        for agent_key, result in agent_results.items():
            parts.append(f"--- {agent_key.upper()} ---")
            if result.get("success"):
                # Extract meaningful output from each agent type
                output = result.get("summary") or result.get("output") or ""
                if not output and result.get("findings"):
                    output = str(result["findings"])
                if not output and result.get("vulnerabilities"):
                    output = str(result["vulnerabilities"])
                if not output and result.get("test_results"):
                    output = (
                        f"Tests: {result.get('passed', 0)} passed, "
                        f"{result.get('failed', 0)} failed "
                        f"(verdict: {result.get('verdict', 'n/a')})"
                    )
                if not output:
                    output = str(result)
                parts.append(output)
            else:
                parts.append(f"Error: {'; '.join(result.get('errors', ['unknown']))}")
            parts.append("")

        return "\n".join(parts)

    # ── Knowledge base helpers ─────────────────────────────────

    async def _retrieve_knowledge_context(self, task: str) -> str:
        """Retrieve relevant knowledge base context for the task."""
        if not self.rag_pipeline:
            return ""
        try:
            result = await self.rag_pipeline.retrieve(task)
            return result.get("context_text", "")
        except Exception as exc:
            self.logger.warning("Knowledge base retrieval failed", error=str(exc))
            return ""

    # ── Memory helpers ─────────────────────────────────────────

    async def _retrieve_past_context(
        self,
        task: str,
        agent_keys: Optional[List[str]] = None,
    ) -> str:
        """
        Retrieve relevant past task experiences from memory.

        Searches episodic memory for similar completed tasks and
        returns a context string that can enrich agent dispatching.
        """
        if not self.memory_manager:
            return ""

        try:
            results = await self.memory_manager.search_memory(
                query=task,
                memory_type=MemoryType.EPISODIC,
                limit=3,
                min_score=0.3,
            )
            if not results:
                return ""

            parts: List[str] = ["Relevant past experiences:"]
            for r in results:
                entry = r.memory
                parts.append(
                    f"- Task: {entry.content} (score: {r.score:.2f})"
                )
            context = "\n".join(parts)
            self.logger.debug(
                "Retrieved past context",
                result_count=len(results),
            )
            return context
        except Exception as exc:
            self.logger.warning("Failed to retrieve past context", error=str(exc))
            return ""

    async def _store_workflow(
        self,
        task: str,
        agents: List[str],
        agent_results: Dict[str, Any],
        final_result: str,
        errors: List[str],
    ) -> None:
        """Store the completed workflow as an episodic memory entry."""
        if not self.memory_manager:
            return

        try:
            success = len(errors) == 0
            score = 1.0 if success else max(0.0, 1.0 - len(errors) * 0.2)

            all_tools: List[str] = []
            for result in agent_results.values():
                all_tools.extend(result.get("tools_used", []))

            episode = EpisodicMemoryEntry(
                task_description=task,
                agents_involved=agents,
                tools_used=list(set(all_tools)),
                result_summary=final_result[:500] if final_result else "",
                errors=errors,
                evaluation_score=score,
                metadata={
                    "agent_count": len(agents),
                    "error_count": len(errors),
                },
            )
            await self.memory_manager.store_episode(episode)
            self.logger.debug("Workflow stored in episodic memory", task=task[:80])
        except Exception as exc:
            self.logger.warning("Failed to store workflow", error=str(exc))

    # ── Public helpers ────────────────────────────────────────

    @property
    def registry(self) -> "AgentRegistry":
        """Return the agent registry."""
        return self._registry

    def get_message_log(self) -> List[Dict[str, Any]]:
        """Return the full message log as serialised dicts."""
        return [m.to_dict() for m in self._message_log]
