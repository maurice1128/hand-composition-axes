"""Where does the compositional signal live -- joint angles, or contact?

Reads the two arms of ``scripts/experiment_contact.sh`` and reports the three
numbers that separate the hypotheses. Both arms share a bundle, seeds and
splits, so every comparison here is paired per seed and the split-to-split
variance that wrecked the original unpaired design cancels.

    penalty(pose | pose-only)      the null already measured on GRAB: ~0
    penalty(pose | contact-aware)  does knowing contact help predict posture?
    penalty(contact)               is the compositional signal in contact?

Readings:

* **penalty(contact) large, penalty(pose) ~0** -- the four dataset nulls are a
  statement about the *representation*, not about hand manipulation. This is
  the outcome worth having, and the one the survey's working explanation
  predicts.
* **both ~0** -- contact does not rescue it either, and "compositional
  difficulty is rare in hand data" stands as measured, with one more
  representation ruled out rather than assumed.
* **penalty(pose | contact-aware) > penalty(pose | pose-only)** -- contact
  carries information about posture that the pose-only model cannot recover.
  Interesting, but not the same claim, and it must not be reported as if it
  were.

    python scripts/analyse_contact.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from analyse_paired import t_test_one_sample  # noqa: E402


def load(run_dir: Path, budget: int) -> dict[int, dict]:
    p = run_dir / "results.json"
    if not p.exists():
        return {}
    rows = [r for r in json.loads(p.read_text(encoding="utf-8"))["results"]
            if r.get("budget") == budget]
    by: dict[int, dict] = {}
    for r in rows:
        by.setdefault(r["seed"], {})[r["kind"]] = r
    return by


def report(name: str, values: np.ndarray) -> None:
    if len(values) < 2:
        print(f"  {name:<46} n={len(values)} -- too few seeds to test")
        return
    _, p = t_test_one_sample(values)
    stars = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
    print(f"  {name:<46} {values.mean():+.5f} +- {values.std(ddof=1):.5f}  "
          f"n={len(values):<3} {(values > 0).sum()}/{len(values)} pos  p={p:.3f}{stars}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pose", default="runs/contact_pose")
    ap.add_argument("--contact", default="runs/contact_full")
    ap.add_argument("--budget", type=int, default=256)
    ap.add_argument("--kind", default="perframe",
                    help="which architecture's penalty to read. The difficulty "
                         "question is about the bandwidth-matched baseline, so "
                         "'perframe' is the default; 'modular' answers whether "
                         "modularity helps once there is difficulty to help with.")
    args = ap.parse_args()

    pose = load(ROOT / args.pose, args.budget)
    full = load(ROOT / args.contact, args.budget)
    if not pose or not full:
        print(f"missing results: pose={len(pose)} seeds, contact={len(full)} seeds")
        return 1

    shared = sorted(set(pose) & set(full))
    if not shared:
        print("the two arms share no seed; they cannot be paired")
        return 1

    p_pose, p_pose_aware, p_contact, p_all = [], [], [], []
    for s in shared:
        a, b = pose[s].get(args.kind), full[s].get(args.kind)
        if a is None or b is None:
            continue
        p_pose.append(a["penalty"])
        p_pose_aware.append(b["penalty"])
        if "penalty_contact" in b:
            p_contact.append(b["penalty_contact"])
            p_all.append(b["penalty_all"])

    print(f"\nGRAB shape->intent, budget {args.budget}, kind {args.kind}, "
          f"{len(p_pose)} shared seeds\n")
    print("Compositional penalty = MSE(never saw composition) - MSE(saw it).")
    print("Positive means the held-out compositions were genuinely hard.\n")
    report("pose channels, pose-only model (27ch)", np.array(p_pose))
    report("pose channels, contact-aware model (43ch)", np.array(p_pose_aware))
    if p_contact:
        report("contact channels, contact-aware model", np.array(p_contact))
        report("all channels, contact-aware model", np.array(p_all))

    if len(p_pose) > 1 and p_contact:
        print()
        diff = np.array(p_contact) - np.array(p_pose)
        report("contact minus pose (paired, same seed)", diff)
        print("\nA positive paired difference means the compositional signal is "
              "\n  measurably present in contact and absent in joint angles --"
              "\n  which is a claim about the representation, not about the datasets.")

    out = ROOT / "runs" / "contact_summary.json"
    out.write_text(json.dumps({
        "seeds": shared, "kind": args.kind, "budget": args.budget,
        "penalty_pose_only": p_pose,
        "penalty_pose_contact_aware": p_pose_aware,
        "penalty_contact": p_contact,
        "penalty_all": p_all,
    }, indent=2), encoding="utf-8")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
