"""Pull named fields out of a workflow's JSON result into files.

Workflow results arrive as one JSON blob with escaped newlines, which is
unreadable in a terminal and impossible to diff. This writes each requested key
to its own file so the drafts become reviewable artifacts in the repo rather
than something that exists only in a task log.

    python scripts/extract_workflow_output.py <task-output.json> related_work=docs/RELATED_WORK.md skeleton=docs/PAPER_SKELETON.md
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path


def load_blob(path: Path) -> dict:
    text = path.read_text(encoding="utf-8", errors="replace")
    # The file may contain surrounding log lines; take the outermost JSON object.
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end <= start:
        raise ValueError(f"no JSON object found in {path}")
    blob = text[start : end + 1]
    try:
        return json.loads(blob)
    except json.JSONDecodeError:
        # Fall back to extracting top-level string fields directly, which
        # survives truncation of later keys.
        out = {}
        for m in re.finditer(r'"(\w+)"\s*:\s*"((?:[^"\\]|\\.)*)"', blob):
            out[m.group(1)] = m.group(2).encode().decode("unicode_escape", "replace")
        if not out:
            raise
        print(f"[extract] JSON was malformed; recovered {len(out)} field(s) by regex")
        return out


def main() -> int:
    if len(sys.argv) < 3:
        print(__doc__)
        return 1
    blob = load_blob(Path(sys.argv[1]))
    # The workflow's return value is nested under "result"; unwrap it, and
    # parse it again if it arrived as an escaped JSON string.
    inner = blob.get("result")
    if isinstance(inner, str):
        try:
            inner = json.loads(inner)
        except json.JSONDecodeError:
            inner = None
    if isinstance(inner, dict):
        blob = {**blob, **inner}
    print(f"[extract] keys present: {sorted(blob)}")

    for spec in sys.argv[2:]:
        key, _, dest = spec.partition("=")
        if key not in blob:
            print(f"[extract] MISSING key {key!r}; skipping {dest}")
            continue
        value = blob[key]
        if not isinstance(value, str):
            value = json.dumps(value, indent=2, ensure_ascii=False)
        out = Path(dest)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(value, encoding="utf-8")
        print(f"[extract] wrote {out}  ({len(value):,} chars)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
