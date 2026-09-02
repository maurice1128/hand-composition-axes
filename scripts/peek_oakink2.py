"""Inspect OakInk2's annotations and its primitive/complex task program.

OakInk2 is the first dataset in this project with compositional structure its
own authors annotated, rather than an axis invented here. Two earlier attempts
used axes chosen from whatever metadata happened to exist -- DexYCB's
subject x object-shape and OakInk-Image's object x intent -- and an audit
flagged that the latter's "category" grouping was the first character of an
object id, sourced to nothing. Every field this script prints is checked
against the files rather than assumed.

What to establish before writing the adapter:
  * where the hand pose lives, in what representation, and at what frame rate;
  * what a "primitive task" is, and how primitives compose into complex tasks;
  * whether the composition is explicit enough to hold out unseen combinations.

    python scripts/peek_oakink2.py D:\\datasets\\oakink2
"""

from __future__ import annotations

import argparse
import json
import pickle
from collections import Counter
from pathlib import Path

import numpy as np


def describe(obj, indent: int = 2, depth: int = 0) -> str:
    pad = " " * indent
    if depth > 3:
        return "..."
    if isinstance(obj, np.ndarray):
        rng = f" range=[{obj.min():.4f}, {obj.max():.4f}]" if obj.size and obj.dtype.kind == "f" else ""
        return f"ndarray{obj.shape} {obj.dtype}{rng}"
    if hasattr(obj, "detach"):
        a = obj.detach().cpu().numpy()
        return f"tensor{tuple(a.shape)} {a.dtype}"
    if isinstance(obj, dict):
        lines = [f"dict({len(obj)})"]
        for k, v in list(obj.items())[:14]:
            lines.append(f"{pad}{k!r}: {describe(v, indent + 2, depth + 1)}")
        if len(obj) > 14:
            lines.append(f"{pad}... {len(obj) - 14} more keys")
        return "\n".join(lines)
    if isinstance(obj, (list, tuple)):
        head = describe(obj[0], indent + 2, depth + 1) if obj else "empty"
        return f"{type(obj).__name__}({len(obj)}) first: {head}"
    return f"{type(obj).__name__} {obj!r}"[:180]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("root", nargs="?", default=r"D:\datasets\oakink2")
    args = ap.parse_args()
    root = Path(args.root)

    anno_dir = root / "anno_preview"
    annos = sorted(anno_dir.glob("*.pkl"))
    print(f"anno_preview: {len(annos)} sequences\n")

    # Camera intrinsics are stored per frame per view and would swamp the
    # output; the hand fields are what the adapter needs.
    NOISY = {"cam_intr", "cam_extr", "cam_def", "cam_selection", "obj_transf",
             "frame_id_list", "mocap_frame_id_list"}

    if annos:
        with annos[0].open("rb") as fh:
            rec = pickle.load(fh)
        print(f"=== {annos[0].name} ===")
        print(f"top-level keys: {sorted(rec)}\n")
        for key in sorted(rec):
            if key in NOISY:
                v = rec[key]
                n = len(v) if hasattr(v, "__len__") else "?"
                print(f"  {key}: (suppressed, {n} entries)")
                continue
            print(f"  {key}: {describe(rec[key], 4, 1)}")
        print()

    # The program directory carries the task structure -- this is the part that
    # makes held-out compositions constructible.
    prog = root / "program" / "program"
    if not prog.exists():
        prog = next((p for p in root.rglob("task_target.json")), Path()).parent
    print(f"=== program directory: {prog} ===")
    if prog.exists():
        for sub in sorted(p for p in prog.iterdir()):
            n = len(list(sub.rglob("*"))) if sub.is_dir() else 1
            print(f"  {sub.name:<28} {'dir' if sub.is_dir() else 'file':>4}  {n} entries")

        tt = prog / "task_target.json"
        if tt.exists():
            data = json.loads(tt.read_text(encoding="utf-8"))
            print(f"\n  task_target.json: {type(data).__name__} with {len(data)} entries")
            for k, v in list(data.items())[:3] if isinstance(data, dict) else enumerate(data[:3]):
                print(f"    {k!r}: {json.dumps(v)[:400]}")

        for name in ("desc_info", "program_info", "pdg", "initial_condition"):
            d = prog / name
            if not d.exists():
                continue
            files = sorted(d.glob("*.json"))
            print(f"\n  --- {name}/  ({len(files)} files) ---")
            for f in files[:2]:
                obj = json.loads(f.read_text(encoding="utf-8"))
                print(f"    {f.name}")
                print("      " + json.dumps(obj, indent=2)[:900].replace("\n", "\n      "))

    # Sequence naming carries scene and subject; count the axes available.
    stems = [a.stem for a in annos]
    scenes = Counter(s.split("__")[0] for s in stems)
    subj = Counter(s.split("__")[1].split("++")[0] for s in stems if "__" in s)
    print(f"\n=== sequence axes ===")
    print(f"  scenes  : {len(scenes)}  {dict(list(scenes.items())[:8])}")
    print(f"  subjects: {len(subj)}  {dict(list(subj.items())[:8])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
