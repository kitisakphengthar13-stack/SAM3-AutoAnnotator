"""Filesystem locations owned by the standalone desktop application."""

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"


def discover_default_model():
    """Return the sole local SAM3 checkpoint, or the preferred named checkpoint."""
    if not MODELS_DIR.is_dir():
        return None
    candidates = sorted(MODELS_DIR.glob("*.pt"))
    preferred = next(
        (path for path in candidates if path.name.casefold().startswith("sam3")),
        None,
    )
    return preferred or (candidates[0] if len(candidates) == 1 else None)
