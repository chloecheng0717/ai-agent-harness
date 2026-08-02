"""
CTF Challenge Registry

Every challenge in the benchmark suite, plus helpers to look them up and
validate them.  Challenges are built fresh each time so their files are
reproducible byte-for-byte.
"""

from __future__ import annotations

from app.benchmarks.challenges.archive import (
    build_archive_nested,
    build_archive_zip,
)
from app.benchmarks.challenges.encoding import (
    build_encoding_base64,
    build_encoding_multilayer,
)
from app.benchmarks.challenges.file_analysis import (
    build_file_analysis_hidden,
    build_file_analysis_strings,
)
from app.benchmarks.challenges.multi_step import build_multi_step
from app.benchmarks.challenges.network import build_perceptron
from app.benchmarks.models import Challenge


def build_all_challenges() -> list[Challenge]:
    """Return a fresh instance of every challenge in the suite."""
    return [
        build_encoding_base64(),
        build_encoding_multilayer(),
        build_archive_zip(),
        build_archive_nested(),
        build_file_analysis_strings(),
        build_file_analysis_hidden(),
        build_multi_step(),
        build_perceptron(),
    ]


def get_challenge(name: str) -> Challenge:
    """Return a fresh challenge by name, raising ``KeyError`` if unknown."""
    by_name = {c.name: c for c in build_all_challenges()}
    return by_name[name]


def challenge_categories() -> list[str]:
    """Distinct challenge categories in suite order."""
    categories: list[str] = []
    for challenge in build_all_challenges():
        if challenge.category not in categories:
            categories.append(challenge.category)
    return categories


#: Convenience: a module-level list of challenge builders (used by scripts).
CHALLENGE_BUILDERS: list[tuple[str, str]] = [
    ("encoding_base64", build_encoding_base64),
    ("encoding_multilayer", build_encoding_multilayer),
    ("archive_zip", build_archive_zip),
    ("archive_nested", build_archive_nested),
    ("file_analysis_strings", build_file_analysis_strings),
    ("file_analysis_hidden", build_file_analysis_hidden),
    ("multi_step", build_multi_step),
    ("perceptron_play_naught", build_perceptron),
]
