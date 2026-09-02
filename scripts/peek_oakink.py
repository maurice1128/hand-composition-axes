"""Inspect the OakInk annotation archive without extracting it.

The archive holds ~1.57M small files; extracting on Windows takes hours and
buys nothing, since Python can read zip members directly. This script dumps the
structure and the shape/dtype of a few samples so the adapter can be written
against what is actually there rather than against the documentation.

    python scripts/peek_oakink.py D:\\datasets\\oakink\\zipped\\image\\anno_v2.1.zip
"""

from __future__ import annotations

import argparse
import io
import json
import pickle
import zipfile
from collections import Counter

import numpy as np


def describe(obj, indent: int = 2) -> str:
    pad = " " * indent
    if isinstance(obj, np.ndarray):
        return f"ndarray shape={obj.shape} dtype={obj.dtype} range=[{obj.min():.4f}, {obj.max():.4f}]"
    if isinstance(obj, dict):
        lines = [f"dict with {len(obj)} keys"]
        for k, v in list(obj.items())[:12]:
            lines.append(f"{pad}{k!r}: {describe(v, indent + 2)}")
        return "\n".join(lines)
    if isinstance(obj, (list, tuple)):
        head = describe(obj[0], indent + 2) if obj else "empty"
        return f"{type(obj).__name__} len={len(obj)}, first: {head}"
    return f"{type(obj).__name__} {obj!r}"[:160]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("archive")
    ap.add_argument("--samples", type=int, default=3)
    args = ap.parse_args()

    with zipfile.ZipFile(args.archive) as zf:
        names = zf.namelist()
        print(f"{len(names):,} entries\n")

        by_dir = Counter(n.rsplit("/", 1)[0] for n in names if not n.endswith("/"))
        for d, c in by_dir.most_common(12):
            print(f"{c:>9,}  {d}")

        # Small top-level json/txt files carry the split definitions.
        print("\n--- small non-pkl entries ---")
        for n in names:
            if not n.endswith((".pkl", "/")) and zf.getinfo(n).file_size < 5_000_000:
                print(f"  {n}  ({zf.getinfo(n).file_size:,} bytes)")

        for folder in ("hand_j", "hand_v", "general_info", "obj_transf"):
            members = [n for n in names if f"/{folder}/" in n and n.endswith(".pkl")]
            if not members:
                continue
            print(f"\n=== anno/{folder} ({len(members):,} files) ===")
            for name in members[: args.samples]:
                with zf.open(name) as fh:
                    obj = pickle.load(io.BytesIO(fh.read()))
                print(f"  {name.rsplit('/', 1)[1]}")
                print(f"    {describe(obj, 6)}")

        # Sequence identity is encoded in the filename; confirm the pattern so
        # frames can be grouped back into trajectories.
        print("\n--- filename parsing ---")
        hand_j = [n for n in names if "/hand_j/" in n and n.endswith(".pkl")]
        stems = [n.rsplit("/", 1)[1][: -len(".pkl")] for n in hand_j[:200000]]
        parts = Counter(len(s.split("__")) for s in stems)
        print(f"  '__'-separated field counts: {dict(parts)}")
        for s in stems[:5]:
            print(f"    {s.split('__')}")
        seqs = Counter("__".join(s.split("__")[:2]) for s in stems)
        print(f"  distinct sequences in first {len(stems):,} files: {len(seqs):,}")
        lens = np.array(sorted(seqs.values()))
        print(
            f"  frames per sequence: min={lens.min()} median={int(np.median(lens))} "
            f"max={lens.max()} mean={lens.mean():.1f}"
        )

        for n in names:
            if n.endswith("seq_status.json"):
                with zf.open(n) as fh:
                    data = json.load(fh)
                print(f"\n--- {n}: {len(data)} entries ---")
                for k, v in list(data.items())[:5]:
                    print(f"  {k!r}: {v!r}"[:200])
                break
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
