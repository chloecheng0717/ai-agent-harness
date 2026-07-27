"""
Unit Tests for AgentMessage

Tests cover:
- Message creation and defaults
- Status transitions
- Serialization
- Metadata handling
"""

import pytest
from datetime import datetime, timezone

from app.communication.agent_message import AgentMessage, MessageStatus


# ── Creation Tests ─────────────────────────────────────────────


class TestAgentMessageCreation:
    def test_minimal_creation(self) -> None:
        msg = AgentMessage(sender="supervisor", receiver="coder", task="write code")
        assert msg.sender == "supervisor"
        assert msg.receiver == "coder"
        assert msg.task == "write code"
        assert msg.response is None
        assert msg.status == MessageStatus.PENDING
        assert msg.metadata == {}
        assert msg.message_id is not None
        assert msg.timestamp is not None

    def test_full_creation(self) -> None:
        msg = AgentMessage(
            sender="supervisor",
            receiver="researcher",
            task="gather info",
            response="done",
            status=MessageStatus.COMPLETED,
            metadata={"priority": "high"},
        )
        assert msg.response == "done"
        assert msg.status == MessageStatus.COMPLETED
        assert msg.metadata == {"priority": "high"}

    def test_unique_ids(self) -> None:
        msg1 = AgentMessage(sender="a", receiver="b", task="t1")
        msg2 = AgentMessage(sender="a", receiver="b", task="t2")
        assert msg1.message_id != msg2.message_id


# ── Status Transition Tests ────────────────────────────────────


class TestMessageStatusTransitions:
    def test_mark_sent(self) -> None:
        msg = AgentMessage(sender="a", receiver="b", task="t")
        msg.mark_sent()
        assert msg.status == MessageStatus.SENT

    def test_mark_processing(self) -> None:
        msg = AgentMessage(sender="a", receiver="b", task="t")
        msg.mark_processing()
        assert msg.status == MessageStatus.PROCESSING

    def test_mark_completed(self) -> None:
        msg = AgentMessage(sender="a", receiver="b", task="t")
        msg.mark_completed("result")
        assert msg.status == MessageStatus.COMPLETED
        assert msg.response == "result"

    def test_mark_failed(self) -> None:
        msg = AgentMessage(sender="a", receiver="b", task="t")
        msg.mark_failed("something broke")
        assert msg.status == MessageStatus.FAILED
        assert msg.response == "something broke"

    def test_full_lifecycle(self) -> None:
        msg = AgentMessage(sender="a", receiver="b", task="t")
        assert msg.status == MessageStatus.PENDING
        msg.mark_sent()
        assert msg.status == MessageStatus.SENT
        msg.mark_processing()
        assert msg.status == MessageStatus.PROCESSING
        msg.mark_completed("done")
        assert msg.status == MessageStatus.COMPLETED


# ── Serialization Tests ────────────────────────────────────────


class TestMessageSerialization:
    def test_to_dict(self) -> None:
        msg = AgentMessage(
            sender="supervisor",
            receiver="qa",
            task="verify",
            metadata={"source": "test"},
        )
        d = msg.to_dict()
        assert d["sender"] == "supervisor"
        assert d["receiver"] == "qa"
        assert d["task"] == "verify"
        assert d["response"] is None
        assert d["status"] == "pending"
        assert d["metadata"] == {"source": "test"}
        assert "message_id" in d
        assert "timestamp" in d

    def test_to_dict_completed(self) -> None:
        msg = AgentMessage(sender="a", receiver="b", task="t")
        msg.mark_completed("response text")
        d = msg.to_dict()
        assert d["response"] == "response text"
        assert d["status"] == "completed"


# ── MessageStatus Enum Tests ───────────────────────────────────


class TestMessageStatusEnum:
    def test_enum_values(self) -> None:
        assert MessageStatus.PENDING.value == "pending"
        assert MessageStatus.SENT.value == "sent"
        assert MessageStatus.PROCESSING.value == "processing"
        assert MessageStatus.COMPLETED.value == "completed"
        assert MessageStatus.FAILED.value == "failed"
