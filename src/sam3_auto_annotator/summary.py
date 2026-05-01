import json


def save_run_summary(path, summary):
    with path.open("w", encoding="utf-8") as summary_file:
        json.dump(summary, summary_file, indent=2)
        summary_file.write("\n")

