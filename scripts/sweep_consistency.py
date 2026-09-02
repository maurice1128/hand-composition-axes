"""Trade-off curve: context consistency versus reconstruction quality.

The measurement in ``scripts/demo_composition.py`` said the learned primitives
are not reusable units -- decoding the same primitive after different
predecessors gives an identity ratio of 0.055, i.e. context explains ~18x more
of the resulting motion than primitive identity. Nothing in the training
objective asked for context-independence, so ``ModularConfig.consistency_weight``
was added to ask for it directly.

A single weight is not a result. Two failure modes bracket the useful range:

* **too small** — at weight 1.0 the consistency term is ~1e-4 against a
  reconstruction term of ~5e-2, five hundred times smaller, so it changes
  nothing and the identity ratio stays where it was;
* **too large** — the cheapest way to make a primitive's motion
  context-independent is to make it produce no motion at all, which satisfies
  the loss perfectly and destroys the model.

So the honest artefact is the curve, not a point: identity ratio and
reconstruction error against weight, with the degenerate branch visible. Where
(and whether) the two can be had together is the actual finding.

    python scripts/sweep_consistency.py --weights 0 10 100 1000
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = ROOT / ".venv" / "Scripts" / "python.exe"


def run(cmd: list[str]) -> str:
    proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    if proc.returncode != 0:
        print(proc.stdout[-2000:])
        print(proc.stderr[-2000:])
        raise RuntimeError(f"command failed: {' '.join(cmd[:4])} ...")
    return proc.stdout


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--weights", type=float, nargs="*", default=[0.0, 10.0, 100.0, 1000.0])
    ap.add_argument("--bundle", default="data/bundles/oakink.npz")
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--contexts", type=int, default=4)
    ap.add_argument("--n-primitives", type=int, default=12)
    ap.add_argument("--out", default="runs/consistency_sweep")
    # Recorded into sweep.json below. The first sweep did not record it, so
    # months later the identity-ratio result could not be attributed to a
    # dataset without reading shell history -- and after the retargeting fix
    # that made the difference between "needs re-running" and "unaffected".
    ap.add_argument("--fresh", action="store_true")
    args = ap.parse_args()

    out_root = ROOT / args.out
    out_root.mkdir(parents=True, exist_ok=True)
    rows = []

    # Repeats of the same weight need distinct directories, or they overwrite
    # each other and a variance check silently becomes a single run.
    seen_counts: dict[float, int] = {}
    for w in args.weights:
        rep = seen_counts.get(w, 0)
        seen_counts[w] = rep + 1
        run_dir = out_root / (f"w{w:g}" if rep == 0 else f"w{w:g}_r{rep}")
        print(f"\n{'=' * 70}\nconsistency_weight = {w:g}  (repeat {rep})\n{'=' * 70}")

        cmd = [
            str(PY), "scripts/train_prior.py", "--model", "modular",
            "--bundle", args.bundle, "--run-dir", str(run_dir),
            "--set", f"train.epochs={args.epochs}", "train.save_every=99999",
            "loader.stride=2", f"model.n_primitives={args.n_primitives}",
            f"model.consistency_weight={w}", f"model.consistency_contexts={args.contexts}",
        ]
        if args.fresh:
            cmd.append("--fresh")
        train_out = run(cmd)
        test_line = [l for l in train_out.splitlines() if l.startswith("[test] loss=")]
        metrics = {}
        if test_line:
            for tok in test_line[0].replace("[test] ", "").split():
                if "=" in tok:
                    k, v = tok.split("=", 1)
                    try:
                        metrics[k] = float(v)
                    except ValueError:
                        pass

        print(run([
            str(PY), "scripts/demo_composition.py",
            "--run-dir", str(run_dir), "--bundle", args.bundle,
        ]).split("--- 2.")[-1][:700])

        comp = json.loads((run_dir / "composition.json").read_text(encoding="utf-8"))
        rows.append({
            "weight": w,
            "recon": metrics.get("recon"),
            "primitives_used": metrics.get("primitives_used"),
            "switches_per_window": metrics.get("switches_per_window"),
            "consistency": metrics.get("consistency"),
            "identity_ratio": comp["identity"]["identity_ratio"],
            "between": comp["identity"]["between_primitive_variance"],
            "within": comp["identity"]["within_primitive_variance"],
            "segmentation_ratio": comp["segmentation"].get("switch_speed_ratio"),
        })
        (out_root / "sweep.json").write_text(
            json.dumps({"args": vars(args), "rows": rows}, indent=2), encoding="utf-8")

    print("\n\n" + "=" * 86)
    print("CONSISTENCY / RECONSTRUCTION TRADE-OFF")
    print("=" * 86)
    print(f"{'weight':>9} {'recon':>9} {'identity':>10} {'between':>10} {'within':>10} "
          f"{'prims':>6} {'seg':>6}  note")
    print("-" * 86)
    base = next((r for r in rows if r["weight"] == 0), None)
    for r in rows:
        note = ""
        if base and base["recon"] and r["recon"]:
            note = f"recon {100 * (r['recon'] / base['recon'] - 1):+.0f}% vs w=0"
        # The degenerate branch: consistency satisfied by producing no motion.
        # Detect it RELATIVE to the unpenalised run, not against a fixed
        # threshold -- an absolute cutoff of 1e-4 failed to fire on a run whose
        # between-primitive variance had collapsed 200x to 8.7e-4, and whose
        # identity ratio therefore looked like a success at 0.995.
        if base and base["between"] and r["weight"] > 0:
            shrink = base["between"] / max(r["between"], 1e-12)
            if shrink > 10:
                note += f"  DEGENERATE: between-primitive variance collapsed {shrink:.0f}x"
        print(
            f"{r['weight']:>9g} {r['recon'] or float('nan'):>9.5f} "
            f"{r['identity_ratio']:>10.3f} {r['between']:>10.6f} {r['within']:>10.6f} "
            f"{r['primitives_used'] or float('nan'):>6.0f} "
            f"{r['segmentation_ratio'] or float('nan'):>6.2f}  {note}"
        )

    print("\nidentity ratio > 1 means primitive identity explains more of the decoded")
    print("motion than the context it sits in -- the property that makes a bank a")
    print("library. Read it together with recon: a ratio bought by destroying")
    print("reconstruction is not a usable prior, and a ratio bought by making every")
    print("primitive motionless is not a prior at all.")
    print(f"\nfull results -> {out_root / 'sweep.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
