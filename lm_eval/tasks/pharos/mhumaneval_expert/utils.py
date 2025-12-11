import re
from typing import Any, Dict


# Extracts the function name from the Python prompt stub.
_FUNC_PATTERN = re.compile(r"def\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\(")


def _add_entry_point(doc: Dict[str, Any]) -> Dict[str, Any]:
    doc = dict(doc)
    doc.setdefault("test", doc.get("tests"))
    match = _FUNC_PATTERN.search(doc.get("prompt", ""))
    if not match:
        raise ValueError("Could not find function signature in prompt.")
    doc["entry_point"] = match.group(1)
    return doc


def process_docs(dataset):
    return dataset.map(_add_entry_point)
