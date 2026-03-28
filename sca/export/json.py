"""JSON export for sca Results."""

from __future__ import annotations

import json

from sca.core.analyzer import Results


def export_json(results: Results, path: str) -> None:
    """
    Serialize Results to a JSON file.

    All numpy arrays are converted to lists for JSON compatibility.
    """
    data = results.to_dict()
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
