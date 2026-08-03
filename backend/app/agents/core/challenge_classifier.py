"""
Challenge Classifier

Detects the type of a CTF task from its goal/description text *before* any
planning happens, so the right strategy is chosen up front:

* A task that points at a remote TCP service (``host:port``, ``nc host
  port``, netcat, socket/connect instructions) is a **network** challenge
  and gets a deterministic socket plan using the ``python`` tool.
* Everything else falls through to the normal (LLM / template) planner.

Network detection deliberately runs FIRST: hostnames such as
``cylabacademy.net`` look like long base64 tokens to naive encoders, which
is exactly why a network task must never be misread as "encoded data to
decode".  Because detection is keyword/regex based it is fully
deterministic and unit-testable without an LLM or a sandbox.
"""

from __future__ import annotations

import re

from app.models.state import PlanStep

# ── Categories ────────────────────────────────────────────────────

NETWORK = "network"
ENCODING = "encoding"
ARCHIVE = "archive"
BINARY = "binary"
WEB = "web"
CRYPTO = "crypto"
UNKNOWN = "unknown"

#: Category names in priority order.  Network must be evaluated first so a
#: hostname is never mistaken for an encoded token.
CATEGORY_PRIORITY = (NETWORK, ARCHIVE, BINARY, WEB, CRYPTO, ENCODING)


# ── Network detection ─────────────────────────────────────────────

#: Strong, unambiguous network-service wording.  Any single hit marks the
#: task as a remote-service interaction.
_STRONG_NETWORK_INDICATORS = (
    "service endpoint",
    "host:port",
    "netcat",
    " nc ",
    "socket",
    "connect to the remote",
    "remote service",
    "tcp connection",
    "interact with the service",
)

#: Weaker signals; two or more must co-occur to count as network.
_WEAK_NETWORK_INDICATORS = (
    "connect",
    "remote",
    "server",
    "port",
    "tcp",
    "send commands",
    "banner",
)

_DOMAIN_RE = r"(?:[a-zA-Z0-9](?:[a-zA-Z0-9\-]{0,61}[a-zA-Z0-9])?\.)+[a-zA-Z]{2,}"
_IPV4_RE = (
    r"(?:25[0-5]|2[0-4]\d|1\d\d|[1-9]?\d)"
    r"(?:\.(?:25[0-5]|2[0-4]\d|1\d\d|[1-9]?\d)){3}"
)
_ENDPOINT_RE = re.compile(
    rf"\b(?P<host>(?:{_DOMAIN_RE})|(?:{_IPV4_RE}))\s*:\s*(?P<port>\d{{2,5}})\b"
)
_NC_RE = re.compile(r"\bnc\s+(?P<host>\S+)\s+(?P<port>\d{2,5})\b")


# ── Other category detection ──────────────────────────────────────

# Tokens embedded in a goal that hint at encoded data.  Lookaround
# assertions (rather than \b) delimit base64 tokens so the optional
# "=" padding stays part of the match.
_B64_PADDED_RE = re.compile(
    r"(?<![A-Za-z0-9+/])[A-Za-z0-9+/]{4,}={1,2}(?![A-Za-z0-9+/=])"
)
_B64_TOKEN_RE = re.compile(
    r"(?<![A-Za-z0-9+/])[A-Za-z0-9+/]{8,}(?![A-Za-z0-9+/=])"
)
_HEX_TOKEN_RE = re.compile(r"\b[0-9a-fA-F]{8,}\b")
_URL_TOKEN_RE = re.compile(r"%[0-9a-fA-F]{2}")

_ARCHIVE_KEYWORDS = (
    "archive",
    "zip",
    "tar",
    "gzip",
    "gz",
    "7z",
    "unzip",
    "decompress",
)
_BINARY_KEYWORDS = (
    "binary",
    "strings",
    "printable",
    "ascii",
    "file type",
    "magic bytes",
    "payload",
)
_WEB_KEYWORDS = ("http", "https", "website", "web page", "curl", "fetch url")
_CRYPTO_KEYWORDS = (
    "cipher",
    "encrypt",
    "decrypt",
    "crypto",
    "rsa",
    "aes",
    "caesar",
    "rot13",
    "vigenere",
)
_ENCODING_KEYWORDS = (
    "base64",
    "hex",
    "url encode",
    "url decode",
    "url-encoded",
    "percent-encode",
    "percent-decode",
    "encoded",
    "decode",
)


# ── Classification ────────────────────────────────────────────────


def classify_challenge(text: str) -> str:
    """
    Return the best-guess challenge category for *text*.

    Categories are: ``network``, ``archive``, ``binary``, ``web``,
    ``crypto``, ``encoding``, or ``unknown``.  Network is checked first so
    a remote-service task is never misread as encoded data, then the
    keyword families, with encoding last because hostnames/IDs otherwise
    trip its token regexes.
    """
    lower = text.lower()

    if is_network_text(lower):
        return NETWORK

    if any(kw in lower for kw in _ARCHIVE_KEYWORDS):
        return ARCHIVE

    if any(kw in lower for kw in _BINARY_KEYWORDS):
        return BINARY

    if any(kw in lower for kw in _WEB_KEYWORDS):
        return WEB

    if any(kw in lower for kw in _CRYPTO_KEYWORDS):
        return CRYPTO

    if any(kw in lower for kw in _ENCODING_KEYWORDS):
        return ENCODING

    if _detect_encoding_token(text) is not None:
        return ENCODING

    return UNKNOWN


def is_network_text(lower_text: str) -> bool:
    """
    Whether *lower_text* (already lowercased) describes a remote service.

    Returns ``True`` when any strong indicator appears, or when at least
    two weak indicators co-occur.  Deliberately excludes the bare word
    "network" on its own so generic goals like "scan the network" are not
    misrouted.
    """
    if any(ind in lower_text for ind in _STRONG_NETWORK_INDICATORS):
        return True
    return (
        sum(1 for ind in _WEAK_NETWORK_INDICATORS if ind in lower_text) >= 2
    )


def _looks_like_base64(token: str) -> bool:
    """
    Whether *token* plausibly is base64 rather than an ordinary word.

    Real base64 text almost always mixes character cases, contains digits,
    or uses ``+``/``/``.  Requiring at least one of these prevents plain
    words (e.g. "describe", "cylabacademy") from being misread as encoded
    tokens.
    """
    has_digit = any(ch.isdigit() for ch in token)
    mixed_case = any(ch.isupper() for ch in token) and any(ch.islower() for ch in token)
    return has_digit or mixed_case or "+" in token or "/" in token


def _detect_encoding_token(text: str) -> str | None:
    """
    Return the encoding family ("base64"/"hex"/"url") for an encoded token
    embedded in *text*, or ``None`` when none is present.
    """
    if _B64_PADDED_RE.search(text):
        return "base64"
    hex_match = _HEX_TOKEN_RE.search(text)
    if hex_match and len(hex_match.group(0)) % 2 == 0:
        return "hex"
    token_match = _B64_TOKEN_RE.search(text)
    if token_match and _looks_like_base64(token_match.group(0)):
        return "base64"
    if _URL_TOKEN_RE.search(text):
        return "url"
    return None


# ── Endpoint extraction ───────────────────────────────────────────


def extract_endpoint(text: str) -> tuple[str, int] | None:
    """
    Extract ``(host, port)`` from *text*, or ``None``.

    Recognises ``nc host port`` commands and ``host:port`` pairs (domain
    names or IPv4 addresses).
    """
    nc_match = _NC_RE.search(text)
    if nc_match:
        return nc_match.group("host"), int(nc_match.group("port"))

    endpoint_match = _ENDPOINT_RE.search(text)
    if endpoint_match:
        return endpoint_match.group("host"), int(endpoint_match.group("port"))

    return None


# ── Deterministic network plan ────────────────────────────────────

#: Python socket code that connects to the service, reads the full banner,
#: saves it to the workspace, and prints it.  The banner carries the
#: training points and protocol, which live on the remote service rather
#: than in local challenge files.
_CONNECT_CODE = """\
import socket
host = {host!r}
port = {port}
s = socket.create_connection((host, port), timeout=15)
s.settimeout(15)
data = b''
while True:
    try:
        chunk = s.recv(4096)
    except socket.timeout:
        break
    if not chunk:
        break
    data += chunk
s.close()
text = data.decode('utf-8', errors='replace')
open('/workspace/banner.txt', 'w').write(text)
print(text)
"""

#: Python code that parses every labeled ``(x, y)`` point from the saved
#: banner and trains a single-layer perceptron (w1, w2, b) so the two
#: classes are separated, persisting the weights for the final step.
_TRAIN_CODE = """\
import json, re
text = open('/workspace/banner.txt').read()
coord = re.compile(r'\\((-?\\d+)\\s*[,;]\\s*(-?\\d+)\\)')
points = []
for line in text.splitlines():
    m = coord.search(line)
    if not m:
        continue
    rest = line[m.end():]
    toks = re.findall(r'[01]', rest)
    if not toks:
        continue
    points.append(((int(m.group(1)), int(m.group(2))), int(toks[0])))
if not points:
    raise SystemExit('no labeled points parsed from banner')
w1, w2, b = 1.0, -1.0, 0.0
lr = 0.1
for _ in range(5000):
    errors = 0
    for (x, y), label in points:
        pred = 1 if w1 * x + w2 * y + b >= 0 else 0
        if pred != label:
            err = label - pred
            w1 += lr * err * x
            w2 += lr * err * y
            b += lr * err
            errors += 1
    if errors == 0:
        break
open('/workspace/weights.json', 'w').write(
    json.dumps({'w1': w1, 'w2': w2, 'b': b})
)
print('trained weights: w1=%s w2=%s b=%s points=%d' % (w1, w2, b, len(points)))
"""

#: Python code that reconnects, submits the trained weights with ``SET``,
#: redraws with ``SHOW``, runs ``CHECK`` and prints the whole response so
#: the service's verification message and flag are captured as evidence.
_CHECK_CODE = """\
import json, socket, time
host = {host!r}
port = {port}
w = json.loads(open('/workspace/weights.json').read())
s = socket.create_connection((host, port), timeout=15)
s.settimeout(20)
banner = s.recv(8192)
s.sendall(('SET %s %s %s\\n' % (w['w1'], w['w2'], w['b'])).encode())
time.sleep(0.3)
s.sendall(b'SHOW\\n')
time.sleep(0.3)
s.sendall(b'CHECK\\n')
data = banner
while True:
    try:
        chunk = s.recv(4096)
    except socket.timeout:
        break
    if not chunk:
        break
    data += chunk
s.close()
text = data.decode('utf-8', errors='replace')
print(text)
"""


def build_network_plan(host: str, port: int) -> list[PlanStep]:
    """
    Build the deterministic, self-contained plan that solves a network
    challenge against the service at ``host:port``.

    Every step uses the ``python`` tool with real socket code; the steps
    share the persistent run workspace so the banner and trained weights
    are carried between them.
    """
    return [
        PlanStep(
            index=0,
            description=(
                f"Connect to the remote service at {host}:{port} and read "
                "the welcome banner"
            ),
            tool_name="python",
            tool_parameters={"code": _CONNECT_CODE.format(host=host, port=port)},
            depends_on=[],
            rationale=(
                "The challenge data lives on a remote TCP service, not in "
                "local files. Open a socket with Python, read the welcome "
                "banner (training points and protocol), and save it for the "
                "next step."
            ),
            priority="high",
            estimated_tools=["python"],
            expected_output=(
                "Welcome banner text including the training points and protocol."
            ),
        ),
        PlanStep(
            index=1,
            description="Parse the training points and train the perceptron weights",
            tool_name="python",
            tool_parameters={"code": _TRAIN_CODE},
            depends_on=[0],
            rationale=(
                "Extract every labeled (x, y) point from the banner and run "
                "the perceptron learning rule to find weights w1, w2 and "
                "bias b that separate the two classes. Save the weights for "
                "the next step."
            ),
            priority="high",
            estimated_tools=["python"],
            expected_output="Trained weights w1, w2, b written to weights.json.",
        ),
        PlanStep(
            index=2,
            description=(
                f"Submit the weights with SET, run SHOW and CHECK on "
                f"{host}:{port}, and read the flag"
            ),
            tool_name="python",
            tool_parameters={"code": _CHECK_CODE.format(host=host, port=port)},
            depends_on=[1],
            rationale=(
                "Reconnect to the service, submit the trained weights with "
                "SET, redraw with SHOW, and run CHECK so the service verifies "
                "every point and prints the flag. Print the full server "
                "response."
            ),
            priority="critical",
            estimated_tools=["python"],
            expected_output=(
                "Server response containing the current weights and the flag."
            ),
        ),
    ]
