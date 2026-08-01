"""
Unit Tests for the Archive Tool

Two test groups (no Docker daemon required):

- :class:`TestArchiveHelpers` exercises the real extraction logic against
  actual archives created on the host (ZIP, TAR, TAR.GZ, GZIP, 7Z).
- :class:`TestArchiveTool` exercises the tool with a mocked ``DockerSandbox``.
"""

import base64
import gzip
import io
import json
import tarfile
import zipfile
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from app.models.tool import (
    RiskLevel,
    SandboxCapability,
    ToolExecutionRequest,
    ToolPermission,
)
from app.sandbox.docker_sandbox import SandboxExecutionResult
from app.tools.archive_helpers import (
    SUPPORTED_FORMATS,
    ArchiveError,
    detect_format,
    extract_archive,
    safe_target,
)
from app.tools.plugins.archive_tool import ArchiveTool
from app.tools.registry import ToolRegistry

# ── Helpers ──────────────────────────────────────────────────────


def _ok_result(
    stdout: str = "",
    stderr: str = "",
    exit_code: int = 0,
) -> SandboxExecutionResult:
    return SandboxExecutionResult(
        success=exit_code == 0,
        exit_code=exit_code,
        stdout=stdout,
        stderr=stderr,
        execution_time=100.0,
        container_id="mock1234",
        image="python:3.12-slim",
        timed_out=False,
    )


def _mock_sandbox(return_value: SandboxExecutionResult | None = None) -> MagicMock:
    sandbox = MagicMock()
    sandbox.execute = AsyncMock(return_value=return_value or _ok_result())
    return sandbox


def _make_zip(
    path: Any,
    files: dict[str, str],
    password: str | None = None,
) -> None:
    with zipfile.ZipFile(path, "w") as archive:
        if password:
            archive.setpassword(password.encode())
        for name, content in files.items():
            archive.writestr(name, content)


# ═══════════════════════════════════════════════════════════════════
# ArchiveHelpers Tests (real extraction, no sandbox)
# ═══════════════════════════════════════════════════════════════════


class TestArchiveHelpers:
    """Unit tests for the shared extraction logic."""

    def test_supported_formats(self) -> None:
        assert SUPPORTED_FORMATS == ("zip", "tar", "tar.gz", "gzip", "7z")

    def test_detect_format_by_extension(self) -> None:
        assert detect_format("a.zip") == "zip"
        assert detect_format("a.tar") == "tar"
        assert detect_format("a.tar.gz") == "tar.gz"
        assert detect_format("a.tgz") == "tar.gz"
        assert detect_format("a.gz") == "gzip"
        assert detect_format("a.gzip") == "gzip"
        assert detect_format("a.7z") == "7z"
        assert detect_format("A.ZIP") == "zip"  # case-insensitive

    def test_detect_format_explicit_override(self) -> None:
        assert detect_format("unknown.dat", explicit="7z") == "7z"
        assert detect_format("unknown.dat", explicit="tar.gz") == "tar.gz"

    def test_detect_format_unknown_extension(self) -> None:
        with pytest.raises(ValueError, match="Unable to detect"):
            detect_format("no_extension")

    def test_detect_format_unsupported_explicit(self) -> None:
        with pytest.raises(ValueError, match="Unsupported archive format"):
            detect_format("a.zip", explicit="rar")

    def test_safe_target_ok(self, tmp_path: Any) -> None:
        target = safe_target(tmp_path, "sub/name.txt")
        assert target == tmp_path / "sub" / "name.txt"

    def test_safe_target_rejects_traversal(self, tmp_path: Any) -> None:
        with pytest.raises(ArchiveError, match="escapes"):
            safe_target(tmp_path, "../evil.txt")

    def test_safe_target_rejects_absolute(self, tmp_path: Any) -> None:
        with pytest.raises(ArchiveError, match="escapes"):
            safe_target(tmp_path, "/etc/passwd")

    def test_extract_zip(self, tmp_path: Any) -> None:
        archive = tmp_path / "data.zip"
        _make_zip(archive, {"flag.txt": "FLAG{zip}", "sub/deep.txt": "deep"})
        result = extract_archive(
            {"archive_path": str(archive), "extract_dir": str(tmp_path / "out")}
        )
        assert result["success"] is True
        assert result["archive_type"] == "zip"
        assert result["file_count"] == 2
        assert set(result["files"]) == {"flag.txt", "sub/deep.txt"}
        assert (tmp_path / "out" / "flag.txt").read_text() == "FLAG{zip}"
        assert (tmp_path / "out" / "sub" / "deep.txt").read_text() == "deep"

    def test_extract_zip_default_destination_is_archive_dir(self, tmp_path: Any) -> None:
        archive = tmp_path / "data.zip"
        _make_zip(archive, {"a.txt": "A"})
        result = extract_archive({"archive_path": str(archive)})
        assert result["success"] is True
        assert result["destination"] == str(tmp_path)
        assert (tmp_path / "a.txt").exists()

    def test_extract_tar(self, tmp_path: Any) -> None:
        archive = tmp_path / "data.tar"
        with tarfile.open(archive, "w") as tf:
            data = b"tar content"
            info = tarfile.TarInfo("inside.txt")
            info.size = len(data)
            tf.addfile(info, io.BytesIO(data))
            dir_info = tarfile.TarInfo("folder")
            dir_info.type = tarfile.DIRTYPE
            tf.addfile(dir_info)
        result = extract_archive(
            {"archive_path": str(archive), "extract_dir": str(tmp_path / "out")}
        )
        assert result["success"] is True
        assert result["archive_type"] == "tar"
        assert result["files"] == ["inside.txt"]
        assert (tmp_path / "out" / "inside.txt").read_bytes() == b"tar content"
        assert (tmp_path / "out" / "folder").is_dir()

    def test_extract_tar_gz(self, tmp_path: Any) -> None:
        archive = tmp_path / "data.tar.gz"
        with tarfile.open(archive, "w:gz") as tf:
            data = b"compressed"
            info = tarfile.TarInfo("g.txt")
            info.size = len(data)
            tf.addfile(info, io.BytesIO(data))
        result = extract_archive(
            {"archive_path": str(archive), "extract_dir": str(tmp_path / "out")}
        )
        assert result["success"] is True
        assert result["archive_type"] == "tar.gz"
        assert result["files"] == ["g.txt"]
        assert (tmp_path / "out" / "g.txt").read_bytes() == b"compressed"

    def test_extract_tar_with_symlink(self, tmp_path: Any) -> None:
        archive = tmp_path / "links.tar"
        with tarfile.open(archive, "w") as tf:
            data = b"real data"
            info = tarfile.TarInfo("real.txt")
            info.size = len(data)
            tf.addfile(info, io.BytesIO(data))
            link = tarfile.TarInfo("link.txt")
            link.type = tarfile.SYMTYPE
            link.linkname = "real.txt"
            tf.addfile(link)
        result = extract_archive(
            {"archive_path": str(archive), "extract_dir": str(tmp_path / "out")}
        )
        assert result["success"] is True
        assert set(result["files"]) == {"real.txt", "link.txt"}
        assert (tmp_path / "out" / "link.txt").read_bytes() == b"real data"

    def test_extract_gzip(self, tmp_path: Any) -> None:
        archive = tmp_path / "message.txt.gz"
        with gzip.open(archive, "wb") as fh:
            fh.write(b"gzip payload")
        result = extract_archive(
            {"archive_path": str(archive), "extract_dir": str(tmp_path / "out")}
        )
        assert result["success"] is True
        assert result["archive_type"] == "gzip"
        assert result["files"] == ["message.txt"]
        assert (tmp_path / "out" / "message.txt").read_bytes() == b"gzip payload"

    def test_extract_gzip_gzip_extension(self, tmp_path: Any) -> None:
        archive = tmp_path / "data.gzip"
        with gzip.open(archive, "wb") as fh:
            fh.write(b"x")
        result = extract_archive(
            {"archive_path": str(archive), "extract_dir": str(tmp_path / "out")}
        )
        assert result["files"] == ["data"]

    def test_extract_from_base64_content(self, tmp_path: Any) -> None:
        source = tmp_path / "src.zip"
        _make_zip(source, {"secret.txt": "FLAG{b64}"})
        encoded = base64.b64encode(source.read_bytes()).decode()
        result = extract_archive(
            {
                "archive_path": str(tmp_path / "uploaded.zip"),
                "content_b64": encoded,
                "extract_dir": str(tmp_path / "out"),
            }
        )
        assert result["success"] is True
        assert (tmp_path / "out" / "secret.txt").read_text() == "FLAG{b64}"

    def test_extract_7z_with_py7zr(self, tmp_path: Any) -> None:
        pytest.importorskip("py7zr")
        import py7zr  # type: ignore[import-not-found]

        archive = tmp_path / "data.7z"
        with py7zr.SevenZipFile(archive, "w") as seven:
            seven.writestr(b"seven content", "seven.txt")
        result = extract_archive(
            {"archive_path": str(archive), "extract_dir": str(tmp_path / "out")}
        )
        assert result["success"] is True
        assert result["archive_type"] == "7z"
        assert result["files"] == ["seven.txt"]
        assert (tmp_path / "out" / "seven.txt").read_bytes() == b"seven content"

    def test_extract_7z_unavailable_reports_gracefully(
        self, tmp_path: Any, monkeypatch: Any
    ) -> None:
        archive = tmp_path / "data.7z"
        archive.write_bytes(b"not a real 7z archive")
        monkeypatch.setattr(
            "app.tools.archive_helpers._get_7z_library", lambda: None
        )
        monkeypatch.setattr(
            "app.tools.archive_helpers.shutil.which", lambda name: None
        )
        result = extract_archive({"archive_path": str(archive)})
        assert result["success"] is False
        assert "7Z support is unavailable" in result["error"]

    def test_corrupted_zip(self, tmp_path: Any) -> None:
        archive = tmp_path / "corrupt.zip"
        archive.write_bytes(b"definitely not a zip file")
        result = extract_archive({"archive_path": str(archive)})
        assert result["success"] is False
        assert "Corrupted or unreadable" in result["error"]

    def test_corrupted_gzip(self, tmp_path: Any) -> None:
        archive = tmp_path / "corrupt.gz"
        archive.write_bytes(b"not gzip data")
        result = extract_archive({"archive_path": str(archive)})
        assert result["success"] is False
        assert result["error"]

    def test_password_protected_zip_without_password(
        self, tmp_path: Any, monkeypatch: Any
    ) -> None:
        archive = tmp_path / "secret.zip"
        _make_zip(archive, {"hidden.txt": "secret"})

        real_infolist = zipfile.ZipFile.infolist

        def fake_infolist(self: zipfile.ZipFile) -> Any:
            infos = real_infolist(self)
            for info in infos:
                info.flag_bits |= 0x1  # mark as encrypted
            return infos

        monkeypatch.setattr(zipfile.ZipFile, "infolist", fake_infolist)
        result = extract_archive({"archive_path": str(archive)})
        assert result["success"] is False
        assert "password-protected" in result["error"].lower()

    def test_path_traversal_zip_slip_rejected(self, tmp_path: Any) -> None:
        archive = tmp_path / "evil.zip"
        _make_zip(archive, {"../evil.txt": "pwned"})
        result = extract_archive(
            {"archive_path": str(archive), "extract_dir": str(tmp_path / "out")}
        )
        assert result["success"] is False
        assert "escapes" in result["error"]
        assert not (tmp_path / "evil.txt").exists()

    def test_path_traversal_tar_slip_rejected(self, tmp_path: Any) -> None:
        archive = tmp_path / "evil.tar"
        with tarfile.open(archive, "w") as tf:
            data = b"pwned"
            info = tarfile.TarInfo("../../evil.txt")
            info.size = len(data)
            tf.addfile(info, io.BytesIO(data))
        result = extract_archive(
            {"archive_path": str(archive), "extract_dir": str(tmp_path / "out")}
        )
        assert result["success"] is False
        assert "escapes" in result["error"]
        assert not (tmp_path / "evil.txt").exists()

    def test_missing_archive(self, tmp_path: Any) -> None:
        result = extract_archive({"archive_path": str(tmp_path / "nope.zip")})
        assert result["success"] is False
        assert "not found" in result["error"]

    def test_no_archive_path(self) -> None:
        result = extract_archive({})
        assert result["success"] is False
        assert "archive_path" in result["error"]

    def test_unsupported_explicit_format(self, tmp_path: Any) -> None:
        archive = tmp_path / "data.zip"
        _make_zip(archive, {"a.txt": "A"})
        result = extract_archive({"archive_path": str(archive), "format": "rar"})
        assert result["success"] is False
        assert "Unsupported archive format" in result["error"]

    def test_forced_format(self, tmp_path: Any) -> None:
        archive = tmp_path / "data.bin"
        _make_zip(archive, {"a.txt": "A"})
        result = extract_archive(
            {
                "archive_path": str(archive),
                "format": "zip",
                "extract_dir": str(tmp_path / "out"),
            }
        )
        assert result["success"] is True
        assert result["archive_type"] == "zip"

    def test_tar_password_rejected(self, tmp_path: Any) -> None:
        archive = tmp_path / "data.tar"
        with tarfile.open(archive, "w") as tf:
            data = b"x"
            info = tarfile.TarInfo("x.txt")
            info.size = len(data)
            tf.addfile(info, io.BytesIO(data))
        result = extract_archive(
            {"archive_path": str(archive), "password": "pw"}
        )
        assert result["success"] is False
        assert "TAR archives do not support passwords" in result["error"]


# ═══════════════════════════════════════════════════════════════════
# ArchiveTool Tests (mocked sandbox)
# ═══════════════════════════════════════════════════════════════════


class TestArchiveTool:
    """Unit tests for ArchiveTool with a mocked DockerSandbox."""

    def test_definition_metadata(self) -> None:
        sandbox = _mock_sandbox()
        tool = ArchiveTool(sandbox)
        assert tool.name == "archive"
        assert "archive" in tool.definition.tags
        assert "ctf" in tool.definition.tags
        assert tool.sandbox_capability == SandboxCapability.CONTAINER
        assert tool.risk_level == RiskLevel.MEDIUM
        assert ToolPermission.WRITE in tool.permissions

    @pytest.mark.asyncio
    async def test_execute_success(self) -> None:
        payload = json.dumps(
            {
                "success": True,
                "files": ["flag.txt"],
                "destination": "/workspace",
                "archive_type": "zip",
                "file_count": 1,
                "error": None,
            }
        )
        sandbox = _mock_sandbox(_ok_result(stdout=payload))
        tool = ArchiveTool(sandbox)
        result = await tool.execute({"archive_path": "/workspace/data.zip"})

        assert result.success is True
        assert result.output["files"] == ["flag.txt"]
        assert result.output["destination"] == "/workspace"
        assert result.output["archive_type"] == "zip"
        assert result.output["file_count"] == 1
        assert result.error is None
        assert result.metadata["archive_type"] == "zip"

    @pytest.mark.asyncio
    async def test_command_and_files_constructed(self) -> None:
        sandbox = _mock_sandbox(
            _ok_result(
                stdout=json.dumps(
                    {"success": True, "files": [], "destination": "/workspace",
                     "archive_type": "zip", "file_count": 0}
                )
            )
        )
        tool = ArchiveTool(sandbox)
        result = await tool.execute(
            {
                "archive_path": "challenge.zip",
                "extract_dir": "out",
                "format": "zip",
                "password": "s3cret",
            }
        )

        assert result.success is True
        cmd = sandbox.execute.call_args[0][0]
        assert "python /workspace/_archive_extract.py" in cmd

        kwargs = sandbox.execute.call_args[1]
        assert kwargs["workspace_id"] == "archivetool-temp"
        files = kwargs["files"]
        assert "_archive_extract.py" in files
        assert "_archive_params.json" in files

        params = json.loads(files["_archive_params.json"])
        assert params["archive_path"] == "/workspace/challenge.zip"
        assert params["extract_dir"] == "/workspace/out"
        assert params["format"] == "zip"
        assert params["password"] == "s3cret"

    @pytest.mark.asyncio
    async def test_content_encoded_as_base64_param(self) -> None:
        sandbox = _mock_sandbox(
            _ok_result(
                stdout=json.dumps(
                    {"success": True, "files": ["a.txt"], "destination": "/workspace",
                     "archive_type": "zip", "file_count": 1}
                )
            )
        )
        tool = ArchiveTool(sandbox)
        result = await tool.execute(
            {"archive_path": "c.zip", "content": "UEsDBA=="}
        )
        assert result.success is True
        kwargs = sandbox.execute.call_args[1]
        params = json.loads(kwargs["files"]["_archive_params.json"])
        assert params["content_b64"] == "UEsDBA=="
        assert params["archive_path"] == "/workspace/c.zip"

    @pytest.mark.asyncio
    async def test_no_input_returns_error(self) -> None:
        sandbox = _mock_sandbox()
        tool = ArchiveTool(sandbox)
        result = await tool.execute({})
        assert result.success is False
        assert result.error is not None
        assert "must be provided" in result.error
        sandbox.execute.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_unsupported_format_returns_error(self) -> None:
        sandbox = _mock_sandbox()
        tool = ArchiveTool(sandbox)
        result = await tool.execute(
            {"archive_path": "a.zip", "format": "rar"}
        )
        assert result.success is False
        assert result.error is not None
        assert "Unsupported archive format" in result.error
        sandbox.execute.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_corrupted_archive_result_propagated(self) -> None:
        payload = json.dumps(
            {
                "success": False,
                "files": [],
                "destination": "",
                "archive_type": None,
                "file_count": 0,
                "error": "Corrupted or unreadable archive: File is not a zip file",
            }
        )
        sandbox = _mock_sandbox(_ok_result(stdout=payload))
        tool = ArchiveTool(sandbox)
        result = await tool.execute({"archive_path": "/workspace/bad.zip"})
        assert result.success is False
        assert result.error is not None
        assert "Corrupted or unreadable" in result.error
        assert result.output["files"] == []

    @pytest.mark.asyncio
    async def test_password_protected_result_propagated(self) -> None:
        payload = json.dumps(
            {
                "success": False,
                "files": [],
                "destination": "",
                "archive_type": None,
                "file_count": 0,
                "error": "ZIP archive is password-protected; provide a 'password' parameter",
            }
        )
        sandbox = _mock_sandbox(_ok_result(stdout=payload))
        tool = ArchiveTool(sandbox)
        result = await tool.execute({"archive_path": "/workspace/enc.zip"})
        assert result.success is False
        assert result.error is not None
        assert "password-protected" in result.error

    @pytest.mark.asyncio
    async def test_sandbox_failure_with_invalid_json(self) -> None:
        sandbox = _mock_sandbox(
            _ok_result(
                stdout="Traceback ...",
                stderr="python: can't open file",
                exit_code=1,
            )
        )
        tool = ArchiveTool(sandbox)
        result = await tool.execute({"archive_path": "/workspace/a.zip"})
        assert result.success is False
        assert result.error is not None
        assert "python: can't open file" in result.error

    @pytest.mark.asyncio
    async def test_sandbox_exception_returns_error(self) -> None:
        sandbox = MagicMock()
        sandbox.execute = AsyncMock(side_effect=RuntimeError("docker down"))
        tool = ArchiveTool(sandbox)
        result = await tool.execute({"archive_path": "/workspace/a.zip"})
        assert result.success is False
        assert result.error is not None
        assert "docker down" in result.error

    @pytest.mark.asyncio
    async def test_helper_source_is_shippable(self) -> None:
        sandbox = _mock_sandbox(
            _ok_result(
                stdout=json.dumps(
                    {"success": True, "files": [], "destination": "/workspace",
                     "archive_type": "zip", "file_count": 0}
                )
            )
        )
        tool = ArchiveTool(sandbox)
        await tool.execute({"archive_path": "a.zip"})
        kwargs = sandbox.execute.call_args[1]
        source = kwargs["files"]["_archive_extract.py"]
        assert "def extract_archive" in source
        assert "if __name__ == \"__main__\":" in source
        assert "__main__" in source

    @pytest.mark.asyncio
    async def test_shared_workspace_used_when_provided(self) -> None:
        from pathlib import Path

        from app.sandbox.workspace import Workspace

        sandbox = _mock_sandbox(
            _ok_result(
                stdout=json.dumps(
                    {"success": True, "files": [], "destination": "/workspace",
                     "archive_type": "zip", "file_count": 0}
                )
            )
        )
        tool = ArchiveTool(sandbox)
        workspace = Workspace(
            workspace_id="ws-test",
            host_path=Path("unused"),
            container_path="/workspace",
        )
        await tool.execute({"archive_path": "a.zip"}, workspace=workspace)
        kwargs = sandbox.execute.call_args[1]
        assert kwargs["workspace"] is workspace
        assert "workspace_id" not in kwargs

    def test_to_container_path(self) -> None:
        tool = ArchiveTool(_mock_sandbox())
        assert tool._to_container_path("rel/path", "/workspace") == "/workspace/rel/path"
        assert tool._to_container_path("/abs/path", "/workspace") == "/abs/path"
        assert tool._to_container_path("", "/workspace") == "/workspace/"

    def test_tool_registry_integration(self) -> None:
        sandbox = _mock_sandbox()
        tool = ArchiveTool(sandbox)
        registry = ToolRegistry()
        registry.register_instance(tool)
        assert registry.get_tool("archive") is tool
        assert registry.get_tool_definition("archive") is tool.definition

    @pytest.mark.asyncio
    async def test_via_registry_execute(self) -> None:
        payload = json.dumps(
            {
                "success": True,
                "files": ["flag.txt"],
                "destination": "/workspace",
                "archive_type": "zip",
                "file_count": 1,
                "error": None,
            }
        )
        sandbox = _mock_sandbox(_ok_result(stdout=payload))
        registry = ToolRegistry()
        registry.register_instance(ArchiveTool(sandbox))
        request = ToolExecutionRequest(
            tool_name="archive",
            parameters={"archive_path": "challenge.zip"},
            agent_permissions=[
                ToolPermission.READ,
                ToolPermission.WRITE,
                ToolPermission.EXECUTE,
            ],
        )
        result = await registry.execute(request)
        assert result.success is True
        assert result.output["file_count"] == 1
