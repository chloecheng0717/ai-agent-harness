"""
Challenge File Builders

Deterministic helpers for constructing challenge payloads (encoded text,
zip/tar archives, and binary blobs) from a plaintext flag.  All builders
are pure functions of their inputs so the same challenge always yields
byte-identical files, which keeps benchmark runs reproducible.
"""

from __future__ import annotations

import base64
import io
import tarfile
import zipfile

FileContent = str | bytes


def b64(s: str) -> str:
    """Base64-encode a string (utf-8), returning the ASCII text."""
    return base64.b64encode(s.encode("utf-8")).decode("ascii")


def hexstr(s: str) -> str:
    """Hex-encode a string (utf-8), returning the ASCII text."""
    return s.encode("utf-8").hex()


def from_b64(text: str) -> str:
    """Decode base64 text back into a string."""
    return base64.b64decode(text).decode("utf-8")


def from_hex(text: str) -> str:
    """Decode hex text back into a string."""
    return bytes.fromhex(text).decode("utf-8")


def make_zip(files: dict[str, FileContent]) -> bytes:
    """Build an in-memory ZIP archive."""
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        for name, content in files.items():
            if isinstance(content, str):
                archive.writestr(name, content.encode("utf-8"))
            else:
                archive.writestr(name, content)
    return buffer.getvalue()


def make_tar_gz(files: dict[str, FileContent]) -> bytes:
    """Build an in-memory TAR.GZ archive."""
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as archive:
        for name, content in files.items():
            raw = content.encode("utf-8") if isinstance(content, str) else content
            info = tarfile.TarInfo(name)
            info.size = len(raw)
            archive.addfile(info, io.BytesIO(raw))
    return buffer.getvalue()


def binary_blob(
    entries: list[bytes],
    magic: bytes = b"\x7fBENCH\x01",
    filler_seed: int = 0,
) -> bytes:
    """
    Build a deterministic "binary" blob around printable *entries*.

    Each entry is embedded as a run of printable ASCII surrounded by
    non-printable filler so ``strings``-style extraction surfaces them
    cleanly while ``file``-style magic sniffing still sees a binary.

    Parameters
    ----------
    entries : list[bytes]
        Printable byte strings to embed (each must stay printable).
    magic : bytes
        Leading magic bytes to make the blob look non-textual.
    filler_seed : int
        Seed for the deterministic filler byte pattern.

    Returns
    -------
    bytes
        The assembled binary blob.
    """
    out = bytearray(magic)
    seed = filler_seed
    for entry in entries:
        # Non-printable filler (0x01..0x1f), deterministic but varied so
        # it never forms printable runs that would merge with the entries.
        filler = bytes(1 + (seed + i * 3) % 0x1f for i in range(48))
        out.extend(filler)
        out.extend(b"\x00\x01\x02")
        out.extend(entry)
        out.extend(b"\x03\x04\x05")
        out.extend(filler)
        seed = (seed + 7) % 256

    out.extend(b"\x00" * 32)
    return bytes(out)


def log_blob(lines: list[str], magic: bytes = b"\x42\x45\x4e\x43") -> bytes:
    """
    Build a "log file" blob: a magic header followed by printable log lines
    separated by newlines, padded with null bytes so it is not pure text.
    """
    body = "\n".join(lines).encode("utf-8")
    return magic + b"\n" + body + b"\n" + b"\x00" * 16
