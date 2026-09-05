import json
import os
import tempfile
from pathlib import Path

from domain import ProjectState


STATE_FILE_NAME = "annotation_state.json"


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
