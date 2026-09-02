"""Emit the primitive-collapse gate as a standalone artifact.

`check_primitive_collapse.py` is the pre-sweep gate: it trains a small bank and
refuses to proceed if the model uses one primitive, which would make the
"modular" arm monolithic in disguise and void every comparison against the
monolithic baseline. It answers a question about a model it trains itself, so it
leaves nothing behind about the models the paper actually reports.

The manuscript therefore states that no standalone artifact is released, and
gives the reading in prose instead: over every K=12 modular model in the sweeps
Table II reports, the minimum final-epoch `val_primitives_used` is 9.732 of 12
and exactly 16 sit below 11.0, all of them OakInk2's.

This script derives that reading from the stored training histories so the claim
has a file behind it. It does not train anything and it does not replace the
gate; it reports what the gate would have seen had it run on the reported models.

Scope is the sweeps behind Table II, listed below. Models are included when the
bank ceiling reached over training is 12, since a K=16 bank is measured against a
different ceiling and the paper puts those outside this scope explicitly.

    python scripts/report_primitive_collapse.py
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

#: The sweeps Table II reports, as named by `plot_difficulty_survey.py`.
TABLE_II = (
    "oakink_official_category", "oakink_category_rest",
    "oakink_official_attr", "oakink_attr_more",
    "oakink_n70_v2", "grab_shape_v2",
    "pc_hard", "pc_easy",
    "oakink2_paired_v2", "dexycb_paired_v2", "oakink_class",
)

#: Below this, a K=12 bank is worth naming. Not a pass threshold: the gate's
#: pass condition is "several primitives in use", and 1.0 is the failure that
#: motivated it. This is the reporting cut the manuscript uses.
NOTABLE = 11.0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="runs/gates/primitive_collapse.txt")
    ap.add_argument("--k", type=int, default=12, help="bank size to report on")
    args = ap.parse_args()

    rows = []
    for h in sorted((ROOT / "runs").glob("*/*/history.json")):
        sweep = h.parent.parent.name
        if sweep not in TABLE_II:
            continue
        try:
            hist = json.loads(h.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        if not isinstance(hist, list) or not hist:
            continue
        final = hist[-1].get("val_primitives_used")
        if final is None:
            continue
        ceiling = max(e.get("val_primitives_used", 0) for e in hist)
        if abs(ceiling - args.k) > 0.1:
            continue
        rows.append((final, sweep, h.parent.name))

    if not rows:
        print("no K=%d models found under the Table II sweeps" % args.k)
        return 1

    rows.sort()
    low = [r for r in rows if r[0] < NOTABLE]
    by_sweep = Counter(r[1] for r in low)

    out = [
        "primitive collapse, derived from stored training histories",
        "scope: every K=%d modular model in the sweeps Table II reports" % args.k,
        "metric: final-epoch val_primitives_used (the gate's own quantity)",
        "",
        "  models examined      %d" % len(rows),
        "  minimum              %.3f of %d" % (rows[0][0], args.k),
        "  below %.1f            %d" % (NOTABLE, len(low)),
        "  those, by sweep      %s" % (dict(by_sweep) or "none"),
        "",
        "  lowest ten:",
    ]
    for v, sweep, run in rows[:10]:
        out.append("    %8.3f  %s/%s" % (v, sweep, run))
    out += [
        "",
        "-" * 58,
    ]
    if rows[0][0] < 2.0:
        out.append("FAIL: a bank collapsed to a single primitive. Any modular/monolithic")
        out.append("comparison drawing on that model is void.")
        verdict = 1
    else:
        out.append("PASS: no bank collapsed. The lowest model still spreads its windows")
        out.append("over %.1f of %d primitives." % (rows[0][0], args.k))
        verdict = 0
    out.append("")
    out.append("This is a report, not the gate. `check_primitive_collapse.py` is the")
    out.append("gate, and it has to pass before a sweep, not after it.")

    dest = ROOT / args.out
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text("\n".join(out) + "\n", encoding="utf-8")
    print("\n".join(out))
    print("\nwrote %s" % dest)
    return verdict


if __name__ == "__main__":
    raise SystemExit(main())
