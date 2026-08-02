"""
Network Challenge: Perceptron Play Naught

A remote interactive service (netcat host:port) that asks the agent to
tune a single-layer perceptron so every 2D training point is classified
correctly.  The points are lifted into 2D specifically so they become
linearly separable; the agent must talk to the service over a socket and
submit the right weights/bias to earn the flag.

Unlike the local file challenges, there is no seed payload to unpack --
the "data" lives behind the remote socket.  The agent needs a tool that
can open a TCP connection and exchange commands (the ``python`` tool with
the standard library ``socket`` module works, and the sandbox network is
enabled for this challenge).
"""

from __future__ import annotations

from app.benchmarks.models import Challenge, Milestone

#: Remote service endpoint for the challenge.
HOST = "aureolin-pixie.cylabacademy.net"
PORT = 63576

#: Service protocol reproduced from the live server (see HELP output).
PROTOCOL = """\
Perceptron Play Naught -- remote perceptron-tuning service.

Goal: classify every labeled 2D point with a single decision boundary
activation = w1*x + w2*y + b.

Classification rule:
    label 0  <=>  w1*x + w2*y + b <  0
    label 1  <=>  w1*x + w2*y + b >= 0

Commands (sent as plain text lines over the TCP socket):
    SET w1 w2 b     set the weights/bias directly (floats are fine)
    ADJUST dw1 dw2 db  add offsets to the current weights
    POINTS          list the training points with their target labels
    SHOW            redraw the ASCII graph and point table
    CHECK           verify every point; prints the flag when perfect
    RESET           back to w1=1.0, w2=-1.0, b=0.0
    HELP / EXIT / QUIT

The server sends its welcome banner (which includes the point table and
the HELP text) immediately after connecting.
"""


def build_perceptron() -> Challenge:
    """Connect to the remote perceptron service and classify every point."""
    flag = "academy{n4ught_bu7_53p4r4b13_277a702d}"
    return Challenge(
        name="perceptron_play_naught",
        category="network",
        description=(
            "A remote single-layer perceptron playground. The agent must "
            "connect to the service, read the labeled training points, "
            "choose weights w1/w2 and bias b so every point is classified "
            "correctly, and submit CHECK to receive the flag."
        ),
        difficulty="easy",
        goal=(
            "Connect to the remote perceptron service at "
            f"{HOST}:{PORT} using Python's socket module (networking is "
            "enabled in the sandbox). Read the welcome banner and the "
            "training points: each point has (x, y) coordinates and a "
            "label (0 or 1). Set the perceptron weights w1, w2 and bias b "
            "so that every point is classified correctly "
            "(label 0 means w1*x + w2*y + b < 0, label 1 means "
            "w1*x + w2*y + b >= 0), then send CHECK to have the service "
            "verify the classification and print the flag. The service "
            "commands are SET w1 w2 b, ADJUST dw1 dw2 db, POINTS, SHOW, "
            "CHECK, and HELP. The full protocol is in /workspace/hint.txt. "
            "Keep interacting with the service until CHECK reports every "
            "point classified correctly and the flag (academy{...}) is "
            "printed."
        ),
        expected_flag=flag,
        required_capabilities=["python"],
        files={
            "hint.txt": (
                f"Service endpoint: {HOST}:{PORT}\n\n{PROTOCOL}"
            )
        },
        milestones=[
            Milestone(
                "Connect to the remote service",
                lambda t: t.succeeded("python") or t.contains_flag(flag),
            ),
            Milestone(
                "Read the training points and labels",
                lambda t: "(+0,-1)" in t.tool_output_text()
                or t.contains_flag(flag),
            ),
            Milestone(
                "Set weights/bias to separate the points",
                lambda t: "Current weights" in t.tool_output_text()
                or t.contains_flag(flag),
            ),
            Milestone(
                "Recover the flag",
                lambda t: t.contains_flag(flag),
            ),
        ],
    )
