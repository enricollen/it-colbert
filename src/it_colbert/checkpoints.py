"""checkpoint discovery for interrupt-and-resume workflows.

these runs are meant to be stopped overnight and picked up the next morning, so
resuming has to be the default path rather than a recovery procedure. the one
thing that makes this fragile is being interrupted *during* a save: the
checkpoint directory exists but is incomplete, and pointing the trainer at it
fails or silently loads a partial model. `latest_checkpoint` therefore validates
before returning, and falls back to the previous good checkpoint.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

logger = logging.getLogger(__name__)

CHECKPOINT_PATTERN = re.compile(r"^checkpoint-(\d+)$")

# a complete transformers checkpoint always has trainer state plus weights
REQUIRED_ANY = (("model.safetensors", "pytorch_model.bin"),)
REQUIRED_ALL = ("trainer_state.json", "optimizer.pt")


def is_complete_checkpoint(path: Path) -> bool:
    """true when a checkpoint directory looks fully written."""
    if not path.is_dir():
        return False
    for group in REQUIRED_ANY:
        if not any((path / name).exists() for name in group):
            return False
    return all((path / name).exists() for name in REQUIRED_ALL)


def list_checkpoints(output_dir: str | Path) -> list[Path]:
    """checkpoint directories under `output_dir`, oldest first by step."""
    root = Path(output_dir)
    if not root.is_dir():
        return []
    found: list[tuple[int, Path]] = []
    for child in root.iterdir():
        match = CHECKPOINT_PATTERN.match(child.name)
        if match and child.is_dir():
            found.append((int(match.group(1)), child))
    return [path for _, path in sorted(found)]


def latest_checkpoint(output_dir: str | Path) -> Path | None:
    """newest complete checkpoint, skipping any truncated by an interrupt."""
    candidates = list_checkpoints(output_dir)
    for path in reversed(candidates):
        if is_complete_checkpoint(path):
            return path
        logger.warning(
            "ignoring incomplete checkpoint %s (interrupted mid-save?)", path
        )
    return None


def resolve_resume(
    output_dir: str | Path,
    explicit: str | None = None,
    auto: bool = True,
) -> str | None:
    """decide what to resume from.

    an explicit path always wins. otherwise, when `auto` is set, pick up the
    newest complete checkpoint so re-running the same command after a stop
    continues instead of starting over.
    """
    if explicit:
        logger.info("resuming from explicit checkpoint %s", explicit)
        return explicit
    if not auto:
        return None
    found = latest_checkpoint(output_dir)
    if found is None:
        logger.info("no checkpoint in %s; starting fresh", output_dir)
        return None
    logger.info("auto-resuming from %s", found)
    return str(found)


def is_stage_finished(output_dir: str | Path) -> bool:
    """true when a training stage already wrote its `final/` model."""
    final = Path(output_dir) / "final"
    if not final.is_dir():
        return False
    return any(
        (final / name).exists()
        for name in ("model.safetensors", "pytorch_model.bin")
    )
