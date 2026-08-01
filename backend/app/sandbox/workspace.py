"""
Persistent Workspace

A shared workspace directory that persists for the lifetime of a single
agent execution run.  Every tool invocation within the run operates on
the same workspace, so files produced by one tool are visible to later
tool calls in the same run.

Each run gets a unique workspace ID, which allows many agent runs to
execute concurrently without sharing or clobbering each other's files.
"""

import shutil
import uuid
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Workspace:
    """
    A persistent, per-execution shared workspace.

    Attributes:
        workspace_id: Unique identifier for the workspace.
        host_path: Host-side path of the workspace directory.
        container_path: Mount point of the workspace inside the container.
    """

    workspace_id: str
    host_path: Path
    container_path: str = "/workspace"

    @classmethod
    def create(
        cls,
        base_dir: str | Path,
        workspace_id: str | None = None,
        container_path: str = "/workspace",
    ) -> "Workspace":
        """
        Create the workspace directory and return a handle to it.

        Args:
            base_dir: Base directory under which workspaces are stored.
            workspace_id: Explicit workspace ID (generated if not given).
            container_path: Where the workspace is mounted in the container.

        Returns:
            A Workspace handle pointing at the created directory.
        """
        ws_id = workspace_id or f"ws-{uuid.uuid4().hex[:12]}"
        host_path = Path(base_dir) / "sandbox_workspaces" / ws_id
        host_path.mkdir(parents=True, exist_ok=True)
        return cls(
            workspace_id=ws_id,
            host_path=host_path,
            container_path=container_path,
        )

    def cleanup(self) -> None:
        """Remove the workspace directory and all of its contents."""
        if self.host_path.exists():
            shutil.rmtree(self.host_path, ignore_errors=True)

    def __str__(self) -> str:
        """String representation for logging."""
        return self.workspace_id
