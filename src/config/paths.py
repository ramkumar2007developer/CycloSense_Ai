"""Project root resolution."""

from __future__ import annotations

from pathlib import Path

from src.exceptions import ConfigurationError

_MARKERS = ("README.md", "requirements.txt", "AGENT_ACTION_LOG.txt")


def find_project_root(start: Path | None = None) -> Path:
    current = (start or Path(__file__)).resolve()
    if current.is_file():
        current = current.parent
    for candidate in (current, *current.parents):
        if (candidate / "src" / "__init__.py").is_file() and any(
            (candidate / m).is_file() for m in _MARKERS
        ):
            return candidate
    raise ConfigurationError("Could not find CycloSense project root.")


def resolve_path(base: Path, value: str | Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path.resolve()
    return (base / path).resolve()
