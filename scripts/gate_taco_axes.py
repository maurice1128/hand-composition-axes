"""Run the pre-training gates on TACO's three candidate axes and apply the fixed rule.

``runs/PREREG_taco_prediction.md`` fixes, before any TACO file was opened, how
the confirmatory axis is chosen: the **first** of action x tool, action x
object, tool x object for which, at the largest held count in {5, 4, 3} that
works, the label-leak gate passes, coverage fails on at most 10 of 40 seeds, the
naive pool exceeds the budget, and the frame-leak check shows at most 1% verbatim
target frames per arm on five seeds. This script runs exactly those gates on
every axis and held count, saves each transcript, and applies that rule and
nothing else. It trains nothing and reads no training result.

The gates are the repo's own scripts, run unmodified via ``runpy`` with their
stdout captured -- one process, because importing torch costs over a minute
here while other jobs hold the CPU, and 27 subprocesses would spend 40 minutes
importing. Each transcript starts with the equivalent command line.

    python scripts/gate_taco_axes.py

What this does NOT do
---------------------
* Passing these gates says a clean split can be built, not that the axis carries
  a penalty, and not that a label describes the motion of the *right* hand.
* The label-leak gate compares labels and the frame-leak gate compares exact
  frame values. Neither detects two takes of the same triplet by the same
  subject that are near- but not bit-identical; the fine-group split is what
  guards that, at the triplet level only.
* The rule's order of preference is a pre-commitment, not a claim that action x
  tool is the most interesting axis. Axes after the selected one are reported
  here as gate results only and would be exploratory if ever run.
* A budget lower than 256 is used only if the smallest naive pool over the 40
  seeds does not exceed 256, and the report says so.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import re
import runpy
import sys
import time
import traceback
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("OMP_NUM_THREADS", "4")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

AXES = ("action_tool", "action_object", "tool_object")  # pre-registered order
HELD = (5, 4, 3)
GRANULARITY = "oakink2_scene_verb"
MAX_COVERAGE_FAILS = 10
MAX_FRAME_LEAK = 0.01


def run_script(script: str, argv: list[str], transcript: Path | None) -> tuple[int | str, str]:
    """Run a repo script as ``__main__`` in this process; return (exit code, stdout)."""
    buf = io.StringIO()
    old_argv = sys.argv
    sys.argv = [script, *argv]
    code: int | str = 0
    try:
        with contextlib.redirect_stdout(buf):
            try:
                runpy.run_path(str(ROOT / script), run_name="__main__")
            except SystemExit as e:
                code = e.code if isinstance(e.code, int) else (0 if e.code is None else 1)
            except Exception as e:  # noqa: BLE001 - a gate that cannot build a split has failed
                code = f"error: {type(e).__name__}: {e}"
                print(traceback.format_exc(), file=buf)
    finally:
        sys.argv = old_argv
    text = buf.getvalue()
    if transcript is not None:
        transcript.parent.mkdir(parents=True, exist_ok=True)
        header = f"$ .venv/Scripts/python.exe {script} {' '.join(argv)}\n"
        transcript.write_text(header + text + f"\n[exit {code}]\n", encoding="utf-8")
    return code, text


def naive_pools(bundle_path: Path, held: int, seeds: list[int], min_chains: int) -> dict:
    import experiment_paired_composition as E
    from caredex.data.base import TrajectoryBundle

    bundle = TrajectoryBundle.load(bundle_path)
    fine = list(bundle.labels)
    bundle.labels = E.coarsen_labels(fine, GRANULARITY)
    sizes, targets, err = [], [], None
    for seed in seeds:
        try:
            s = E.build_paired_split(bundle, held, seed, fine_labels=fine, min_chains=min_chains)
        except ValueError as e:
            err = str(e)
            break
        sizes.append(len(s["naive_pool"]))
        targets.append(len(s["target"]))
    return {"min": min(sizes) if sizes else None, "max": max(sizes) if sizes else None,
            "target_min": min(targets) if targets else None,
            "target_max": max(targets) if targets else None, "error": err}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bundle-dir", default="data/bundles")
    ap.add_argument("--budget", type=int, default=256)
    ap.add_argument("--min-chains", type=int, default=4)
    ap.add_argument("--min-per-composition", type=int, default=5)
    ap.add_argument("--seeds", type=int, nargs="*", default=list(range(40)))
    ap.add_argument("--frame-seeds", type=int, nargs="*", default=[0, 1, 2, 3, 4])
    ap.add_argument("--out", default="runs/taco_gates.json")
    args = ap.parse_args()

    gates = ROOT / "runs" / "gates"
    seeds = [str(s) for s in args.seeds]
    rows = []
    t0 = time.time()
    for axis in AXES:
        rel = f"{args.bundle_dir}/taco_{axis}.npz"
        for held in HELD:
            pools = naive_pools(ROOT / rel, held, args.seeds, args.min_chains)
            budget, note = args.budget, ""
            if pools["min"] is not None and pools["min"] <= budget:
                budget = 128
                note = (f"naive pool min {pools['min']} does not exceed {args.budget}; "
                        f"gates run at budget 128")
            common = ["--bundle", rel, "--granularity", GRANULARITY,
                      "--held-compositions", str(held), "--min-chains", str(args.min_chains),
                      "--min-per-composition", str(args.min_per_composition)]

            leak_code, _ = run_script(
                "scripts/check_composition_leak.py",
                [*common, "--budget", str(budget), "--seeds", *seeds],
                gates / f"composition_leak_taco_{axis}_h{held}.txt")
            cov_code, cov_text = run_script(
                "scripts/check_informed_coverage.py",
                [*common, "--budgets", str(budget), "--seeds", *seeds],
                gates / f"cover_taco_{axis}_h{held}.txt")
            ok_rows = len(re.findall(r"\sok\s*$", cov_text, flags=re.M))
            thin_rows = len(re.findall(r"TOO THIN\s*$", cov_text, flags=re.M))

            fl_out = f"runs/frame_leak_taco_{axis}_h{held}.json"
            fl_code, fl_text = run_script(
                "scripts/check_frame_leak_b64.py",
                ["--case", f"taco_{axis}={rel}", "--granularity", GRANULARITY,
                 "--held-compositions", str(held), "--min-chains", str(args.min_chains),
                 "--min-per-composition", str(args.min_per_composition),
                 "--budget", str(budget), "--seeds", *[str(s) for s in args.frame_seeds],
                 "--out", fl_out],
                gates / f"frame_leak_taco_{axis}_h{held}.txt")
            worst = None
            if (ROOT / fl_out).exists() and isinstance(fl_code, int):
                worst = json.loads((ROOT / fl_out).read_text(encoding="utf-8"))[
                    f"taco_{axis}"]["max_any_seed_any_arm"]

            scored = ok_rows + thin_rows
            qualifies = (
                leak_code == 0
                and scored == len(args.seeds) and thin_rows <= MAX_COVERAGE_FAILS
                and pools["min"] is not None and pools["min"] > budget
                and worst is not None and worst <= MAX_FRAME_LEAK
            )
            row = {
                "axis": axis, "held": held, "budget": budget, "budget_note": note,
                "label_leak_exit": leak_code, "coverage_exit": cov_code,
                "coverage_rows_scored": scored, "coverage_failures": thin_rows,
                "frame_leak_exit": fl_code, "frame_leak_worst": worst,
                "naive_pool": pools, "qualifies": bool(qualifies),
            }
            rows.append(row)
            print(f"{axis:<14} h{held} budget {budget}: leak exit {leak_code!s:<8.8} "
                  f"coverage {thin_rows}/{scored} fail  frame leak "
                  f"{'n/a' if worst is None else f'{100 * worst:.2f}%'}  "
                  f"naive pool {pools['min']}-{pools['max']}  "
                  f"{'QUALIFIES' if qualifies else 'no'}  [{(time.time() - t0) / 60:.1f} min]",
                  flush=True)

    # The rule: first axis in the fixed order; within it, the largest held count.
    selected = next((r for axis in AXES for r in rows
                     if r["axis"] == axis and r["qualifies"]), None)
    out = {
        "written": time.strftime("%Y-%m-%d %H:%M"),
        "rule": "runs/PREREG_taco_prediction.md: first axis in order action_tool, action_object, "
                "tool_object that qualifies, at the largest held count in 5,4,3",
        "settings": {"granularity": GRANULARITY, "min_chains": args.min_chains,
                     "min_per_composition": args.min_per_composition,
                     "seeds": args.seeds, "frame_seeds": args.frame_seeds},
        "rows": rows,
        "selected": ({"axis": selected["axis"], "held": selected["held"],
                      "budget": selected["budget"]} if selected else None),
    }
    (ROOT / args.out).write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(f"\nrule selects: {out['selected']}\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
