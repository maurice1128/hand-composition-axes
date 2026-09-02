"""How much compositional structure does OakInk2 actually contain?

Every compositional experiment so far has been limited by the axis available in
the data. DexYCB's subject x object-shape grid produced a paired penalty of
+0.00006 -- literally nothing to generalise over -- because each clip is one
grasp of one object. OakInk-Image was little better, and the "category" axis
used there turned out to be the first character of an object id, invented here
and sourced to nothing.

OakInk2 annotates named primitives with explicit frame ranges, so for the first
time the composition axis comes from the dataset's authors. Before writing an
adapter, this establishes whether the axis is dense enough to hold out
transitions:

  * how many distinct primitives, and how often each occurs;
  * how many distinct ordered transitions, and how many sequences carry each;
  * how many sequences contain more than one primitive at all -- a sequence
    with one primitive contributes no transition and cannot test composition.

The number that decides feasibility is trajectories-per-transition. OakInk-Image
had 2.23 per object->intent cell, which was too thin to split into a target and
an informed set; the paired design needs several.

    python scripts/survey_oakink2_primitives.py
"""

from __future__ import annotations

import argparse
import ast
import json
import sys
from collections import Counter
from pathlib import Path


def parse_span(key: str) -> tuple[tuple | None, tuple | None]:
    """Keys look like ``"(None, (966, 5954))"`` -- (left-hand span, right-hand span)."""
    try:
        val = ast.literal_eval(key)
    except (ValueError, SyntaxError):
        return None, None
    if isinstance(val, tuple) and len(val) == 2:
        return val[0], val[1]
    return None, None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", default=r"D:\datasets\oakink2")
    ap.add_argument("--hand", choices=("rh", "lh", "any"), default="rh")
    args = ap.parse_args()

    prog = Path(args.root) / "program" / "program" / "program_info"
    if not prog.exists():
        print(f"not found: {prog}")
        return 1

    files = sorted(prog.glob("*.json"))
    print(f"{len(files)} sequences with primitive annotation\n")

    prim_counts: Counter[str] = Counter()
    trans_counts: Counter[tuple[str, str]] = Counter()
    seqs_per_transition: Counter[tuple[str, str]] = Counter()
    segs_per_seq: list[int] = []
    multi = 0
    seg_lengths: list[int] = []

    key_field = {"rh": "primitive_rh", "lh": "primitive_lh", "any": "primitive"}[args.hand]
    span_index = {"rh": 1, "lh": 0, "any": 1}[args.hand]

    for f in files:
        data = json.loads(f.read_text(encoding="utf-8"))
        segs = []
        for key, val in data.items():
            prim = val.get(key_field)
            if not prim:
                continue
            spans = parse_span(key)
            span = spans[span_index] or spans[1] or spans[0]
            if not span:
                continue
            segs.append((int(span[0]), int(span[1]), str(prim)))

        segs.sort()
        segs_per_seq.append(len(segs))
        for a, b, _ in segs:
            seg_lengths.append(b - a)
        for _, _, p in segs:
            prim_counts[p] += 1
        if len(segs) > 1:
            multi += 1
            local = set()
            for (_, _, p), (_, _, q) in zip(segs[:-1], segs[1:]):
                trans_counts[(p, q)] += 1
                local.add((p, q))
            for t in local:
                seqs_per_transition[t] += 1

    print(f"=== primitives ({args.hand}) ===")
    print(f"distinct: {len(prim_counts)}   total segments: {sum(prim_counts.values())}")
    for p, c in prim_counts.most_common():
        print(f"  {p:<16} {c:>5}")

    n_seq = len(files)
    print(f"\n=== sequence structure ===")
    print(f"  sequences with >1 primitive segment: {multi} / {n_seq} ({multi / max(n_seq, 1):.0%})")
    if segs_per_seq:
        srt = sorted(segs_per_seq)
        print(f"  segments per sequence: min {srt[0]}  median {srt[len(srt) // 2]}  max {srt[-1]}")
    if seg_lengths:
        srt = sorted(seg_lengths)
        print(f"  segment length (mocap frames): min {srt[0]}  median {srt[len(srt) // 2]}  max {srt[-1]}")

    print(f"\n=== transitions (the composition axis) ===")
    print(f"  distinct ordered transitions: {len(trans_counts)}")
    print(f"  possible given {len(prim_counts)} primitives: {len(prim_counts) ** 2}")
    if seqs_per_transition:
        per = sorted(seqs_per_transition.values())
        dense = sum(1 for v in per if v >= 6)
        print(f"  sequences per transition: min {per[0]}  median {per[len(per) // 2]}  max {per[-1]}")
        print(f"  transitions occurring in >= 6 sequences: {dense} / {len(seqs_per_transition)}")
        print(f"\n  most common:")
        for (p, q), c in seqs_per_transition.most_common(12):
            print(f"    {p} -> {q:<14} {c:>4} sequences")

        # OakInk-Image's object->intent axis had 2.23 trajectories per cell and
        # could not be split into target and informed sets.
        avg = sum(seqs_per_transition.values()) / max(len(seqs_per_transition), 1)
        print(f"\n  mean sequences per transition: {avg:.2f}")
        print("  " + (
            f"FEASIBLE: {dense} transitions have >= 6 sequences, enough to split into "
            "target and informed sets."
            if dense >= 4 else
            "TOO THIN: fewer than 4 transitions have >= 6 sequences; a paired split "
            "would leave the informed model barely informed, as happened with OakInk-Image."
        ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
