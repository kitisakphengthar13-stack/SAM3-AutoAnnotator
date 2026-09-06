import json
import os
import tempfile
from pathlib import Path

from domain import ProjectState


STATE_FILE_NAME = "annotation_state.json"
RECOVERY_FILE_NAME = "annotation_state.recovery.json"


def save_project_state(project_state, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        text=True,
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as state_file:
            json.dump(project_state.to_dict(), state_file, indent=2)
            state_file.write("\n")
            state_file.flush()
            os.fsync(state_file.fileno())
        os.replace(temporary_path, path)
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise
    return path


def load_project_state(path):
    path = Path(path)
    with path.open("r", encoding="utf-8") as state_file:
        return ProjectState.from_dict(json.load(state_file))


def recovery_path(output_dir):
    return Path(output_dir) / RECOVERY_FILE_NAME


def save_recovery_state(project_state, output_dir):
    return save_project_state(project_state, recovery_path(output_dir))


def clear_recovery_state(output_dir):
    path = recovery_path(output_dir)
    path.unlink(missing_ok=True)
    return path


def newer_recovery_for(state_path):
    state_path = Path(state_path)
    candidate = state_path.with_name(RECOVERY_FILE_NAME)
    if not candidate.is_file():
        return None
    if not state_path.exists() or candidate.stat().st_mtime_ns > state_path.stat().st_mtime_ns:
        return candidate
    return None
