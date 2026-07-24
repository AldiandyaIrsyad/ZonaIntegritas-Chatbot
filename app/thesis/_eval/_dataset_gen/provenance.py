"""Provenance sidecars for generated evaluation subsets.

Every generated subset gets a ``<output>.meta.json`` recording which panel,
generator, and settings produced it.

Why this exists: the committed subsets carry no record of how they were built.
``logs/`` holds no dataset-generation logs, and the CSVs have no metadata
columns, so the panel that validated ``data/subset_a.csv`` cannot be
reconstructed from the repository — it can only be asserted from memory. That
matters because the panel has changed more than once (see the swap history in
``config.py``), which makes "all subsets were validated by the same ≥4/5 panel"
an unverifiable claim in the thesis rather than a checkable one.

The sidecar is written next to the CSV and is cheap enough to produce on every
run, including aborted ones.
"""

from __future__ import annotations

import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

import structlog

from app.thesis._eval._dataset_gen.config import DatasetGenSettings

logger = structlog.get_logger(__name__)


def _git_sha() -> Optional[str]:
    """Return the current git commit SHA, or None outside a repo.

    Returns:
        The short SHA, or None if git is unavailable or this is not a checkout.
    """
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout.strip() or None if result.returncode == 0 else None


def write_provenance(
    output_path: str,
    subset: str,
    settings: DatasetGenSettings,
    row_count: int,
    extra: Optional[Dict[str, Any]] = None,
) -> Path:
    """Write a ``.meta.json`` sidecar describing how a subset was generated.

    Args:
        output_path: Path of the subset CSV this describes. The sidecar is
            written alongside it with the ``.csv`` suffix replaced.
        subset: Subset identifier, e.g. "a", "b", "c", "d", "d_hard".
        settings: The settings the run actually used — recorded verbatim, so a
            sidecar reflects the live ``.env`` rather than the code defaults.
        row_count: Number of rows written to the CSV.
        extra: Per-builder fields (sampling seed, per-document caps, source
            subsets, KB document count, ...).

    Returns:
        Path to the written sidecar.
    """
    meta: Dict[str, Any] = {
        "subset": subset,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_sha": _git_sha(),
        "row_count": row_count,
        "generator": {
            "model": settings.generator_model,
            "temperature": settings.generator_temperature,
        },
        "panel": {
            "models": settings.panel_model_list,
            "size": len(settings.panel_model_list),
            "acceptance_threshold": settings.acceptance_threshold,
            "temperature": settings.panel_temperature,
        },
        "output_csv": str(Path(output_path).name),
    }
    if extra:
        meta.update(extra)

    sidecar = Path(output_path).with_suffix(".meta.json")
    sidecar.parent.mkdir(parents=True, exist_ok=True)
    sidecar.write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    logger.info(
        "datagen.provenance_written",
        path=str(sidecar),
        subset=subset,
        panel_size=len(settings.panel_model_list),
        rows=row_count,
    )
    return sidecar
