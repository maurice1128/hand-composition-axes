"""Is any degree of freedom dead, pinned or saturated in a bundle?

This gate exists because the one that was supposed to catch this did not.
``limit_clip_fraction`` documents itself as measuring values "outside their
limits **before** clamping", but three of four adapters called it on the output
of ``clamp_to_limits``. It therefore measured the clamp, not the conversion, and
returned 0.00% for a dataset in which ``thumb_mcp_abd`` sat at exactly its lower
limit in every single frame.

Nothing numerical exposed that. It was found by rendering a hand and noticing
the thumb was a straight spike, which is a bad way to find bugs -- so this
checks the property directly:

* **pinned** -- fraction of frames at (within tolerance of) a limit. A real
  joint touches its limit occasionally; a broken conversion sits there.
* **dead** -- normalised standard deviation near zero. A DOF that never moves
  carries no information, and if the models are being compared on how well they
  predict 27 numbers, a constant one quietly dilutes every error metric.
* **range use** -- how much of the anatomical box the data actually visits.

Run it on every bundle before trusting anything downstream of one.

    python scripts/check_dof_health.py
    python scripts/check_dof_health.py --bundle data/bundles/grab.npz --verbose
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from caredex.data.base import TrajectoryBundle  # noqa: E402
from caredex.data.mano_retarget import MANO_UNAVAILABLE_DOF  # noqa: E402
from caredex.hand_model import DOF_NAMES, LIMITS_HI, LIMITS_LO, normalize  # noqa: E402

#: A DOF at a limit for more than this fraction of frames is reported. Real
#: joints do touch limits -- fingers fully extend -- so this is deliberately
#: not zero. The failure this was written for sat at 100%.
PINNED_FRACTION = 0.25

#: Normalised sd below this counts as dead. The limit box is 2 units wide, so
#: 0.01 is 0.5% of the range: a joint moving less than that is a constant.
DEAD_SD = 0.01


def audit(path: Path, tol: float = 1e-4) -> tuple[list[dict], dict]:
    b = TrajectoryBundle.load(path)
    q = np.concatenate([np.asarray(t) for t in b.trajectories])
    x = normalize(q)

    rows = []
    for i, name in enumerate(DOF_NAMES):
        lo_hit = float(np.mean(q[:, i] <= LIMITS_LO[i] + tol))
        hi_hit = float(np.mean(q[:, i] >= LIMITS_HI[i] - tol))
        span = LIMITS_HI[i] - LIMITS_LO[i]
        rows.append({
            "dof": name,
            "at_lo": lo_hit,
            "at_hi": hi_hit,
            "pinned": lo_hit + hi_hit,
            "sd": float(x[:, i].std()),
            "range_used": float(q[:, i].max() - q[:, i].min()) / span if span > 0 else 0.0,
        })
    return rows, {
        "n_frames": len(q),
        "source": b.meta.get("source", path.stem),
        "mano_derived": "flexion_axis" in b.meta,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bundle", nargs="*", default=None)
    ap.add_argument("--verbose", action="store_true", help="print every DOF, not just problems")
    args = ap.parse_args()

    paths = ([ROOT / b for b in args.bundle] if args.bundle
             else sorted((ROOT / "data" / "bundles").glob("*.npz")))
    failed = False

    for p in paths:
        if not p.exists():
            print(f"skip {p}: not built")
            continue
        rows, meta = audit(p)
        # Wrist rotation is not recoverable from MANO -- there is no forearm --
        # so a MANO-derived bundle having three constant wrist DOF is expected,
        # not a defect. Reporting it as failure every run would train the reader
        # to ignore this gate, which is how the last one stopped working.
        expected_dead = set(MANO_UNAVAILABLE_DOF) if meta["mano_derived"] else set()
        bad = [r for r in rows
               if r["dof"] not in expected_dead
               and (r["pinned"] > PINNED_FRACTION or r["sd"] < DEAD_SD)]

        print(f"\n{p.name}  ({meta['n_frames']:,} frames)")
        show = rows if args.verbose else bad
        if not bad:
            print("  PASS: no DOF pinned or dead.")
        for r in sorted(show, key=lambda r: -r["pinned"]):
            flags = []
            if r["pinned"] > PINNED_FRACTION:
                flags.append(f"PINNED {r['pinned']:.0%} (lo {r['at_lo']:.0%} / hi {r['at_hi']:.0%})")
            if r["sd"] < DEAD_SD:
                flags.append(f"DEAD sd={r['sd']:.4f}")
            tag = "  ".join(flags) or f"ok  sd={r['sd']:.3f}"
            print(f"  {r['dof']:<20} range used {r['range_used']:>5.1%}   {tag}")
        if bad:
            failed = True
            print(f"  FAIL: {len(bad)} of {len(rows)} DOF unusable. "
                  f"Anything measured on this bundle is measured on {len(rows) - len(bad)} DOF.")

    print()
    if failed:
        print("At least one bundle has a dead or pinned DOF. Fix the retargeting, "
              "do not widen the limits -- the limits are anatomy.")
        return 1
    print("All bundles pass.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
