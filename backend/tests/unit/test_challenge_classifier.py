"""
Unit Tests for ChallengeClassifier and deterministic network planning

Tests cover:
- classify_challenge categories (network, archive, binary, web, crypto,
  encoding, unknown)
- Network-before-encoding priority
- is_network_text strong/weak indicators
- extract_endpoint variants (nc, host:port, IPv4, none)
- build_network_plan structure and content
- PlannerAgent routing network goals to the deterministic socket plan
- Orchestrator _build_network_plan override and sandbox gating
- Executing the socket plan end-to-end against a scripted service
"""

import contextlib
import io
import socket as socket_module
from pathlib import Path

import pytest
from app.agents.core.challenge_classifier import (
    ARCHIVE,
    BINARY,
    CRYPTO,
    ENCODING,
    NETWORK,
    UNKNOWN,
    WEB,
    build_network_plan,
    classify_challenge,
    extract_endpoint,
    is_network_text,
)
from app.agents.core.orchestrator import Orchestrator
from app.agents.specialized.planner import PlannerAgent
from app.models.state import AgentState

# ── classify_challenge ────────────────────────────────────────────


class TestClassifyChallenge:
    def test_network_nc_command(self) -> None:
        text = (
            "Title: Perceptron Play Naught\n"
            "$ nc aureolin-pixie.cylabacademy.net 63576\n"
        )
        assert classify_challenge(text) == NETWORK

    def test_network_service_endpoint(self) -> None:
        text = "Service endpoint: challenge.example.com:12345"
        assert classify_challenge(text) == NETWORK

    def test_network_socket_instructions(self) -> None:
        text = "Connect to the remote service via TCP. Follow its commands."
        assert classify_challenge(text) == NETWORK

    def test_archive(self) -> None:
        text = "Extract the zip archive and find the hidden note inside."
        assert classify_challenge(text) == ARCHIVE

    def test_binary(self) -> None:
        text = "The payload binary has hidden printable text. Use strings."
        assert classify_challenge(text) == BINARY

    def test_web(self) -> None:
        text = "Fetch the URL https://example.com/page and inspect it."
        assert classify_challenge(text) == WEB

    def test_crypto(self) -> None:
        text = "Decrypt the ciphertext produced with AES and recover the flag."
        assert classify_challenge(text) == CRYPTO

    def test_encoding_keyword(self) -> None:
        text = "Decode the base64 string and then the hex layer underneath."
        assert classify_challenge(text) == ENCODING

    def test_encoding_token(self) -> None:
        text = "The flag was hidden as aVZhbHVlPXllcw== in the file."
        assert classify_challenge(text) == ENCODING

    def test_unknown(self) -> None:
        text = "Greet the user and describe the day."
        assert classify_challenge(text) == UNKNOWN

    def test_network_wins_over_encoding_tokens(self) -> None:
        # The hostname looks like a long base64 token; network must win.
        text = (
            "Connect: nc service.cylabacademy.net 12345 "
            "the token aVZhbHVlPXllcw== is the password"
        )
        assert classify_challenge(text) == NETWORK

    def test_network_wins_over_hex_tokens(self) -> None:
        text = "Use netcat to reach 10.0.0.5:8080. secret = 0xdeadbeefcafe"
        assert classify_challenge(text) == NETWORK

    def test_bare_network_word_not_network(self) -> None:
        text = "Scan the network and report open ports."
        assert classify_challenge(text) == UNKNOWN


# ── is_network_text ───────────────────────────────────────────────


class TestIsNetworkText:
    def test_strong_indicator_alone(self) -> None:
        assert is_network_text("use netcat to connect")
        assert is_network_text("service endpoint is x")

    def test_two_weak_indicators(self) -> None:
        assert is_network_text("connect to the server on port 80")
        assert is_network_text("remote banner on tcp")

    def test_single_weak_indicator_alone(self) -> None:
        assert not is_network_text("the network is down")
        assert not is_network_text("connect the dots")

    def test_empty_text(self) -> None:
        assert not is_network_text("")


# ── extract_endpoint ──────────────────────────────────────────────


class TestExtractEndpoint:
    def test_nc_domain(self) -> None:
        assert extract_endpoint("$ nc aureolin-pixie.cylabacademy.net 63576") == (
            "aureolin-pixie.cylabacademy.net",
            63576,
        )

    def test_nc_ipv4(self) -> None:
        assert extract_endpoint("nc 10.0.0.5 8080") == ("10.0.0.5", 8080)

    def test_host_colon_port(self) -> None:
        assert extract_endpoint("reach challenge.example.com:12345") == (
            "challenge.example.com",
            12345,
        )

    def test_ipv4_colon_port(self) -> None:
        assert extract_endpoint("10.0.0.5:8080 is the target") == ("10.0.0.5", 8080)

    def test_no_endpoint(self) -> None:
        assert extract_endpoint("decode this base64 aGVsbG8=") is None
        assert extract_endpoint("") is None

    def test_port_out_of_range_not_matched(self) -> None:
        assert extract_endpoint("port 99999 is invalid") is None


# ── build_network_plan ────────────────────────────────────────────


class TestBuildNetworkPlan:
    def test_three_python_steps(self) -> None:
        plan = build_network_plan("host.example.com", 1234)
        assert len(plan) == 3
        for step in plan:
            assert step.tool_name == "python"
            assert step.description
            assert step.expected_output
            assert step.priority in {"high", "critical"}

    def test_dependencies_are_topological(self) -> None:
        plan = build_network_plan("host.example.com", 1234)
        assert plan[0].depends_on == []
        assert plan[1].depends_on == [0]
        assert plan[2].depends_on == [1]

    def test_code_embeds_endpoint(self) -> None:
        plan = build_network_plan("svc.cylabacademy.net", 63576)
        assert "svc.cylabacademy.net" in plan[0].tool_parameters["code"]
        assert "63576" in plan[0].tool_parameters["code"]
        assert "svc.cylabacademy.net" in plan[2].tool_parameters["code"]
        assert "63576" in plan[2].tool_parameters["code"]

    def test_plan_passes_validation(self) -> None:
        planner = PlannerAgent()
        plan = build_network_plan("host.example.com", 1234)
        assert planner.validate_plan(plan) == []

    def test_training_and_check_code_present(self) -> None:
        plan = build_network_plan("host.example.com", 1234)
        code2 = plan[1].tool_parameters["code"]
        code3 = plan[2].tool_parameters["code"]
        assert "perceptron" not in code2  # no comment required, just behavior
        assert "SET" in code3
        assert "CHECK" in code3


# ── PlannerAgent routing ──────────────────────────────────────────


class TestPlannerNetworkRouting:
    @pytest.mark.asyncio
    async def test_network_goal_returns_deterministic_plan(self) -> None:
        agent = PlannerAgent()
        result = await agent.process(
            {
                "goal": "$ nc svc.cylabacademy.net 63576 and solve the challenge",
                "tools_available": ["python"],
            }
        )
        assert result["success"] is True
        assert result["category"] == NETWORK
        assert result["plan_source"] == "network_detected"
        assert len(result["plan"]) == 3
        assert all(step["tool_name"] == "python" for step in result["plan"])

    @pytest.mark.asyncio
    async def test_network_goal_without_python_falls_back(self) -> None:
        agent = PlannerAgent()
        result = await agent.process(
            {
                "goal": "$ nc svc.cylabacademy.net 63576",
                "tools_available": ["file", "strings"],
            }
        )
        # No python tool available -> deterministic plan not used.
        assert result.get("plan_source") != "network_detected"

    @pytest.mark.asyncio
    async def test_non_network_goal_uses_normal_planning(self) -> None:
        agent = PlannerAgent()
        result = await agent.process(
            {
                "goal": "Decode the hex token 0xdeadbeef",
                "tools_available": ["python", "encoding"],
            }
        )
        assert result.get("category") != NETWORK
        assert result.get("plan_source") != "network_detected"


# ── Orchestrator _build_network_plan ──────────────────────────────


class TestOrchestratorNetworkOverride:
    def _state_with_user_text(self, text: str) -> AgentState:
        state = AgentState()
        state.goal = text
        state.tools_available = ["python", "file", "strings"]
        state.add_message("user", text)
        return state

    @pytest.mark.asyncio
    async def test_network_plan_built_from_goal(self) -> None:
        orch = Orchestrator()  # no sandbox -> network enabled
        state = self._state_with_user_text(
            "$ nc svc.cylabacademy.net 4444 and grab the flag"
        )
        state = await orch.create_plan(state)
        assert state.metadata.get("plan_source") == "network_detected"
        assert state.metadata.get("challenge_category") == NETWORK
        assert len(state.plan) == 3

    @pytest.mark.asyncio
    async def test_non_network_goal_untouched(self) -> None:
        orch = Orchestrator()
        state = self._state_with_user_text("decode base64 aGVsbG8=")
        state = await orch.create_plan(state)
        assert state.metadata.get("plan_source") != "network_detected"

    @pytest.mark.asyncio
    async def test_network_disabled_sandbox_blocks_override(self) -> None:
        from typing import cast

        from app.sandbox.docker_sandbox import DockerSandbox, SandboxConfig

        class FakeSandbox:
            def __init__(self) -> None:
                self.config = SandboxConfig(network_disabled=True)

        orch = Orchestrator()
        orch.sandbox = cast(DockerSandbox, FakeSandbox())
        state = self._state_with_user_text(
            "$ nc svc.cylabacademy.net 4444 and grab the flag"
        )
        state = await orch.create_plan(state)
        assert state.metadata.get("plan_source") != "network_detected"

    @pytest.mark.asyncio
    async def test_network_enabled_sandbox_allows_override(self) -> None:
        from typing import cast

        from app.sandbox.docker_sandbox import DockerSandbox, SandboxConfig

        class FakeSandbox:
            def __init__(self) -> None:
                self.config = SandboxConfig(network_disabled=False)

        orch = Orchestrator()
        orch.sandbox = cast(DockerSandbox, FakeSandbox())
        state = self._state_with_user_text(
            "$ nc svc.cylabacademy.net 4444 and grab the flag"
        )
        state = await orch.create_plan(state)
        assert state.metadata.get("plan_source") == "network_detected"


# ── Executing the network plan against a scripted service ─────────


class _FakeSocket:
    """Minimal socket stand-in that hands back scripted responses."""

    def __init__(self, responses: list[bytes]) -> None:
        self._responses = list(responses)
        self.sent: list[bytes] = []

    def settimeout(self, timeout: float) -> None:
        pass

    def recv(self, bufsize: int) -> bytes:
        return self._responses.pop(0) if self._responses else b""

    def sendall(self, data: bytes) -> None:
        self.sent.append(data)

    def close(self) -> None:
        pass


class _FakeSocketModule:
    """Replaces ``socket`` for plan code: no real TCP is attempted."""

    def __init__(self, sockets: list[_FakeSocket]) -> None:
        self._sockets = list(sockets)
        self.timeout = TimeoutError

    def create_connection(
        self, address: tuple[str, int], timeout: float = 15
    ) -> _FakeSocket:
        assert self._sockets, f"unexpected connection attempt to {address}"
        return self._sockets.pop(0)


BANNER = """\
Perceptron Play Naught -- remote perceptron-tuning service.
Training points (x, y) -> label:
  (-2,-2) -> 0
  (-1,-1) -> 0
  (1,1) -> 1
  (2,2) -> 1
Commands: SET w1 w2 b | POINTS | SHOW | CHECK
"""


class TestNetworkPlanExecution:
    def test_three_steps_recover_flag(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        connect_sock = _FakeSocket([BANNER.encode()])
        check_sock = _FakeSocket(
            [
                BANNER.encode(),
                b"Current weights: w1=1.0 w2=1.0 b=0.0\n",
                b"All points classified correctly. flag{perceptron-solved}\n",
            ]
        )
        fake_socket = _FakeSocketModule([connect_sock, check_sock])

        plan = build_network_plan("svc.example.com", 4444)
        assert len(plan) == 3

        # Step 0: connect and save the banner to the workspace.
        _run_step_code(plan[0].tool_parameters["code"], fake_socket, tmp_path, monkeypatch)
        assert (tmp_path / "banner.txt").read_text() == BANNER

        # Step 1: train weights; training opens no new connections, so the
        # check-step socket is still queued untouched.
        _run_step_code(plan[1].tool_parameters["code"], fake_socket, tmp_path, monkeypatch)
        assert fake_socket._sockets == [check_sock]

        # Step 2: submit SET/SHOW/CHECK and read the flag from the reply.
        output = _run_step_code(plan[2].tool_parameters["code"], fake_socket, tmp_path, monkeypatch)
        assert "SET" in check_sock.sent[0].decode()
        assert b"SHOW\n" in check_sock.sent
        assert b"CHECK\n" in check_sock.sent
        assert "flag{perceptron-solved}" in output


def _run_step_code(
    code: str,
    socket_holder: _FakeSocketModule,
    workspace: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> str:
    """Execute one plan step's python code with /workspace remapped to the
    host workspace and a scripted socket module in place of the real one.
    Returns whatever the step printed to stdout."""
    monkeypatch.setattr(
        socket_module, "create_connection", socket_holder.create_connection
    )
    remapped = code.replace("/workspace", workspace.as_posix())
    buffer = io.StringIO()
    ns: dict[str, object] = {}
    with contextlib.redirect_stdout(buffer):
        exec(compile(remapped, "<network-plan-step>", "exec"), ns)
    return buffer.getvalue()
