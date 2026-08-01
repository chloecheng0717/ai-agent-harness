"""
Archive Extraction Helpers

Pure extraction logic shared between the :class:`ArchiveTool` (host side)
and the self-contained helper script that runs inside the Docker sandbox.

The tool ships the source of this module into the sandbox as
``_archive_extract.py`` and executes it, so the code that extracts archives
in production is byte-for-byte the code exercised by the unit tests.

Supported formats: ZIP, TAR, TAR.GZ, GZIP, and 7Z.

Security: archive member names are validated against path-traversal
("zip-slip") attacks; members that escape the extraction directory are
rejected.
"""

from __future__ import annotations

import base64
import gzip
import importlib
import json
import os
import shutil
import subprocess
import tarfile
import zipfile
from pathlib import Path
from typing import Any

# Canonical archive format names understood by the tool and helper.
SUPPORTED_FORMATS = ("zip", "tar", "tar.gz", "gzip", "7z")


class ArchiveError(Exception):
    """Raised when an archive cannot be extracted for a known reason."""


def detect_format(archive_path: str | Path, explicit: str | None = None) -> str:
    """
    Detect the archive format.

    Parameters
    ----------
    archive_path : str | Path
        Path of the archive file.
    explicit : str | None
        Force a format instead of detecting it from the file extension.

    Returns
    -------
    str
        Canonical format name (see :data:`SUPPORTED_FORMATS`).

    Raises
    ------
    ValueError
        If the format cannot be detected or is unsupported.
    """
    fmt = explicit.strip().lower() if explicit else None
    if fmt is not None:
        if fmt not in SUPPORTED_FORMATS:
            raise ValueError(
                f"Unsupported archive format '{explicit}'. "
                f"Supported: {', '.join(SUPPORTED_FORMATS)}"
            )
        return fmt

    lower = str(archive_path).lower()
    if lower.endswith(".tar.gz") or lower.endswith(".tgz"):
        return "tar.gz"

    suffix = Path(lower).suffix
    mapping = {
        ".zip": "zip",
        ".tar": "tar",
        ".gz": "gzip",
        ".gzip": "gzip",
        ".7z": "7z",
    }
    fmt = mapping.get(suffix)
    if fmt is None:
        raise ValueError(
            f"Unable to detect archive format from '{archive_path}'. "
            f"Supported extensions: {', '.join(sorted(mapping))}. "
            "Pass 'format' explicitly."
        )
    return fmt


def safe_target(dest_root: Path, member_name: str) -> Path:
    """
    Resolve an archive member name inside ``dest_root`` safely.

    Rejects absolute paths and ``..`` traversal (zip-slip).  The check is
    purely lexical (no disk access), so it behaves identically on POSIX
    sandboxes and Windows host test runners, including under 8.3 short
    path names.

    Parameters
    ----------
    dest_root : Path
        Extraction directory the member must stay within.
    member_name : str
        Member name as stored in the archive.

    Returns
    -------
    Path
        The member's destination path.

    Raises
    ------
    ArchiveError
        If the member would escape the extraction directory.
    """
    target = Path(dest_root) / member_name
    normalized_target = os.path.normcase(os.path.abspath(str(target)))
    normalized_root = os.path.normcase(os.path.abspath(str(dest_root)))
    try:
        common = os.path.commonpath([normalized_target, normalized_root])
    except ValueError:
        # Different Windows drives -> cannot be inside dest_root.
        raise ArchiveError(
            f"Archive member escapes the extraction directory: {member_name!r}"
        ) from None
    if common != normalized_root:
        raise ArchiveError(
            f"Archive member escapes the extraction directory: {member_name!r}"
        )
    return target


def _rel_name(target: Path, dest: Path) -> str:
    """Relative path with forward slashes for consistent cross-platform output."""
    return str(target.relative_to(dest)).replace(os.sep, "/")


def _write_member(tf: tarfile.TarFile, member: tarfile.TarInfo, target: Path) -> bool:
    """Extract a regular file member to ``target``, creating parents."""
    source = tf.extractfile(member)
    if source is None:
        return False
    target.parent.mkdir(parents=True, exist_ok=True)
    with source, open(target, "wb") as out:
        shutil.copyfileobj(source, out)
    return True


def _extract_zip(path: Path, dest: Path, password: str | None) -> list[str]:
    """Extract a ZIP archive, resolving passwords for encrypted entries."""
    dest.mkdir(parents=True, exist_ok=True)
    files: list[str] = []
    with zipfile.ZipFile(str(path)) as archive:
        if password:
            archive.setpassword(password.encode("utf-8"))
        for info in archive.infolist():
            if info.is_dir():
                continue
            if info.flag_bits & 0x1 and not password:
                raise ArchiveError(
                    "ZIP archive is password-protected; "
                    "provide a 'password' parameter"
                )
            target = safe_target(dest, info.filename)
            target.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(info) as src, open(target, "wb") as out:
                shutil.copyfileobj(src, out)
            files.append(_rel_name(target, dest))
    return files


def _extract_tar(path: Path, dest: Path, password: str | None) -> list[str]:
    """Extract a TAR or TAR.GZ archive."""
    if password:
        raise ArchiveError("TAR archives do not support passwords")
    dest.mkdir(parents=True, exist_ok=True)
    files: list[str] = []
    with tarfile.open(str(path), mode="r:*") as archive:
        members = archive.getmembers()
        for member in members:
            safe_target(dest, member.name)
        for member in members:
            if member.isdir():
                safe_target(dest, member.name).mkdir(parents=True, exist_ok=True)
                continue
            if member.isfile():
                target = safe_target(dest, member.name)
                if _write_member(archive, member, target):
                    files.append(_rel_name(target, dest))
                continue
            if member.issym() or member.islnk():
                target = safe_target(dest, member.name)
                # Resolve link targets to an in-archive member when possible so
                # extraction works even where creating real symlinks is not
                # permitted (e.g. unprivileged Windows hosts).
                try:
                    link_member = archive.getmember(member.linkname)
                except KeyError:
                    link_member = None
                if link_member is not None and link_member.isfile():
                    if _write_member(archive, link_member, target):
                        files.append(_rel_name(target, dest))
                else:
                    resolved = safe_target(target.parent, member.linkname)
                    if resolved.is_file():
                        target.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copyfile(resolved, target)
                        files.append(_rel_name(target, dest))
    return files


def _gzip_output_name(path: Path) -> str:
    """Derive the decompressed filename from a ``.gz``/``.gzip`` archive."""
    name = path.name
    lower = name.lower()
    if lower.endswith(".gz") and len(name) > 3:
        return name[:-3]
    if lower.endswith(".gzip") and len(name) > 5:
        return name[:-5]
    return path.stem


def _extract_gzip(path: Path, dest: Path, password: str | None) -> list[str]:
    """Decompress a single-member GZIP archive into ``dest``."""
    if password:
        raise ArchiveError("GZIP files do not support passwords")
    dest.mkdir(parents=True, exist_ok=True)
    with gzip.open(str(path), "rb") as fh:
        raw = fh.read()
    target = safe_target(dest, _gzip_output_name(path))
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(raw)
    return [_rel_name(target, dest)]


def _get_7z_library() -> Any | None:
    """Return the ``py7zr`` module if available, else ``None``."""
    try:
        return importlib.import_module("py7zr")
    except ImportError:
        return None


def _extract_7z(path: Path, dest: Path, password: str | None) -> list[str]:
    """Extract a 7Z archive via ``py7zr`` or the system ``7z`` binary."""
    dest.mkdir(parents=True, exist_ok=True)

    py7zr = _get_7z_library()

    if py7zr is not None:
        seven_zip_file = getattr(py7zr, "SevenZipFile", None)
        if seven_zip_file is None:
            raise ArchiveError("py7zr is installed but unusable")
        try:
            with seven_zip_file(str(path), mode="r", password=password) as archive:
                names = [str(n) for n in archive.getnames()]
                for name in names:
                    safe_target(dest, name)
                archive.extractall(path=str(dest))
        except Exception as exc:
            msg = str(exc)
            if "password" in msg.lower():
                raise ArchiveError(
                    "7Z archive is password-protected; "
                    "provide a 'password' parameter"
                ) from exc
            raise ArchiveError(f"7Z extraction failed: {msg}") from exc
        return sorted(
            _rel_name(p, dest) for p in dest.rglob("*") if p.is_file()
        )

    if shutil.which("7z") is None:
        raise ArchiveError(
            "7Z support is unavailable in the sandbox image "
            "(py7zr or the '7z' binary is required)"
        )

    command = ["7z", "x", "-y", "-o" + str(dest), str(path)]
    if password:
        command.append("-p" + password)
    proc = subprocess.run(command, capture_output=True, text=True)
    if proc.returncode != 0:
        output = (proc.stderr + "\n" + proc.stdout).lower()
        if "password" in output or "encrypted" in output:
            raise ArchiveError(
                "7Z archive is password-protected or the password is wrong; "
                "provide the correct 'password' parameter"
            )
        raise ArchiveError(
            f"7Z extraction failed (exit {proc.returncode}): "
            f"{(proc.stderr or proc.stdout).strip()[:500]}"
        )
    return sorted(
        _rel_name(p, dest) for p in dest.rglob("*") if p.is_file()
    )


def _failure(message: str) -> dict[str, Any]:
    """Build a structured failure result."""
    return {
        "success": False,
        "files": [],
        "destination": "",
        "archive_type": None,
        "file_count": 0,
        "error": message,
    }


def extract_archive(params: dict[str, Any]) -> dict[str, Any]:
    """
    Extract an archive according to *params*.

    Parameters
    ----------
    params : dict
        Supported keys:

        - ``archive_path`` (str, required): path of the archive.
        - ``content_b64`` (str, optional): base64-encoded archive bytes that
          are written to ``archive_path`` before extraction.
        - ``extract_dir`` (str, optional): destination directory.  Defaults to
          the archive's parent directory.
        - ``format`` (str, optional): force a format (see
          :data:`SUPPORTED_FORMATS`).  Auto-detected from the extension when
          omitted.
        - ``password`` (str, optional): password for encrypted ZIP / 7Z files.

    Returns
    -------
    dict
        Structured result with ``success``, ``files``, ``destination``,
        ``archive_type``, ``file_count``, and ``error`` keys.
    """
    archive_path = params.get("archive_path")
    if not archive_path:
        return _failure("No 'archive_path' provided")

    archive_path = Path(archive_path)

    content_b64 = params.get("content_b64")
    if content_b64:
        archive_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            archive_path.write_bytes(base64.b64decode(content_b64))
        except Exception as exc:
            return _failure(f"Failed to decode archive content: {exc}")

    if not archive_path.is_file():
        return _failure(f"Archive file not found: {archive_path}")

    try:
        fmt = detect_format(archive_path, params.get("format"))
    except ValueError as exc:
        return _failure(str(exc))

    extract_dir = params.get("extract_dir") or str(archive_path.parent)
    dest = Path(extract_dir)
    dest.mkdir(parents=True, exist_ok=True)
    password = params.get("password") or None

    try:
        if fmt == "zip":
            files = _extract_zip(archive_path, dest, password)
        elif fmt in ("tar", "tar.gz"):
            files = _extract_tar(archive_path, dest, password)
        elif fmt == "gzip":
            files = _extract_gzip(archive_path, dest, password)
        elif fmt == "7z":
            files = _extract_7z(archive_path, dest, password)
        else:
            return _failure(f"Unsupported archive format: {fmt}")
    except ArchiveError as exc:
        return _failure(str(exc))
    except (zipfile.BadZipFile, tarfile.TarError, EOFError, OSError, ValueError) as exc:
        return _failure(f"Corrupted or unreadable archive: {exc}")
    except Exception as exc:
        return _failure(f"Extraction failed: {exc}")

    return {
        "success": True,
        "files": sorted(files),
        "destination": str(dest),
        "archive_type": fmt,
        "file_count": len(files),
        "error": None,
    }


def _cleanup_helper_files(params_path: Path) -> None:
    """Remove the helper script and params file from the workspace."""
    script_path = Path(__file__).resolve()
    for path in (params_path, script_path):
        try:
            if path.exists():
                path.unlink()
        except OSError:
            pass


def main() -> None:
    """
    Sandbox entrypoint.

    Reads ``_archive_params.json`` from the same directory as this script,
    extracts the archive, prints a single JSON result to stdout, and cleans
    the helper files up afterwards.
    """
    params_path = Path(__file__).resolve().parent / "_archive_params.json"
    try:
        with open(params_path, encoding="utf-8") as fh:
            params = json.load(fh)
        result = extract_archive(params)
    except Exception as exc:
        result = _failure(f"Unexpected error: {exc}")
    finally:
        _cleanup_helper_files(params_path)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
