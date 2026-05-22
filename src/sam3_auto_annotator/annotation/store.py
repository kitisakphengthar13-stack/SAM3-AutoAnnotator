import json
from pathlib import Path

from sam3_auto_annotator.annotation.models import ProjectState


STATE_FILE_NAME = "annotation_state.json"


def save_project_state(project_state, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as state_file:
        json.dump(project_state.to_dict(), state_file, indent=2)
        state_file.write("\n")
    return path


def load_project_state(path):
    path = Path(path)
    with path.open("r", encoding="utf-8") as state_file:
        return ProjectState.from_dict(json.load(state_file))
