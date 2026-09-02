"""Turn the data-efficiency sweep into the figure the hypothesis lives or dies on.

Three panels:

1. **Compositional gap vs data budget** — the headline. The hypothesis predicts
   the modular curve stays low while the monolithic curve rises as data shrinks.
2. **Compositional ratio** (unseen / seen) — the same comparison with overall
   model quality divided out. This panel matters because a modular model that is
   simply better everywhere would win panel 1 without the compositional story
   being true. If the advantage survives here, it is about composition.
3. **Absolute errors** — the sanity panel. If either model is not learning at
   all, panels 1 and 2 are ratios of noise.

Bars are annotated with the number of primitives actually used. A run with
``primitives_used == 1`` is drawn hatched and must not be read as evidence
either way: the bank collapsed and the "modular" model was monolithic.

    python scripts/plot_data_efficiency.py runs/de_synth runs/de_oakink
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

COLORS = {"monolithic": "#B45309", "modular": "#1D4ED8"}
MARKERS = {"monolithic": "o", "modular": "s"}


def load(run_dir: Path) -> dict:
    p = run_dir / "results.json"
    if not p.exists():
        raise FileNotFoundError(f"no results at {p}")
    return json.loads(p.read_text(encoding="utf-8"))


def series(results: list[dict], kind: str, key: str) -> tuple[list[float], list[float]]:
    rows = sorted((r for r in results if r["kind"] == kind), key=lambda r: r["frac"])
    return [r["frac"] for r in rows], [r[key] for r in rows]


def plot_run(data: dict, title: str, out: Path) -> Path:
    results = data["results"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))

    for kind in ("monolithic", "modular"):
        x, y = series(results, kind, "compositional_gap")
        axes[0].plot(x, y, marker=MARKERS[kind], color=COLORS[kind], label=kind, lw=2)
        x, y = series(results, kind, "compositional_ratio")
        axes[1].plot(x, y, marker=MARKERS[kind], color=COLORS[kind], label=kind, lw=2)
        for key, ls in (("mse_test_seen", "-"), ("mse_test_unseen", "--")):
            x, y = series(results, kind, key)
            axes[2].plot(
                x, y, marker=MARKERS[kind], color=COLORS[kind], ls=ls, lw=1.6,
                label=f"{kind} {'seen' if 'seen' in key and 'un' not in key else 'unseen'}",
            )

    axes[0].set_title("compositional gap\nMSE(unseen) - MSE(seen)")
    axes[0].set_ylabel("gap (normalised units)")
    axes[1].set_title("compositional ratio\nMSE(unseen) / MSE(seen)")
    axes[1].set_ylabel("ratio")
    axes[1].axhline(1.0, color="0.6", lw=1, ls=":")
    axes[2].set_title("absolute reconstruction error")
    axes[2].set_ylabel("MSE (normalised units)")

    for ax in axes:
        ax.set_xscale("log")
        ax.set_xlabel("fraction of training trajectories")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)

    # Flag any collapsed run: those points are not interpretable.
    collapsed = [r for r in results if (r.get("primitives_used") or 99) <= 1.5]
    if collapsed:
        fig.text(
            0.5, 0.005,
            f"WARNING: {len(collapsed)} modular run(s) collapsed to a single primitive "
            "— those points are not evidence about modularity",
            ha="center", color="#B91C1C", fontsize=9,
        )

    fig.suptitle(title, fontsize=13)
    fig.tight_layout(rect=(0, 0.03, 1, 0.96))
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return out


def verdict(data: dict) -> str:
    """State plainly whether the hypothesis held, including when it did not."""
    results = data["results"]
    fracs = sorted({r["frac"] for r in results})
    lines = []
    advantages = []

    for f in fracs:
        row = {r["kind"]: r for r in results if r["frac"] == f}
        if len(row) < 2:
            continue
        m, d = row["monolithic"], row["modular"]
        if (d.get("primitives_used") or 99) <= 1.5:
            lines.append(f"  frac={f:<7g} VOID — modular bank collapsed")
            continue
        adv = m["compositional_gap"] - d["compositional_gap"]
        advantages.append((f, adv))
        lines.append(
            f"  frac={f:<7g} monolithic {m['compositional_gap']:.5f}  "
            f"modular {d['compositional_gap']:.5f}  "
            f"advantage {adv:+.5f}  ({'modular' if adv > 0 else 'monolithic'} wins)"
        )

    out = ["compositional gap by budget:", *lines, ""]
    if len(advantages) < 2:
        out.append("VERDICT: not enough interpretable budgets to judge the trend.")
        return "\n".join(out)

    advantages.sort()
    low, high = advantages[0], advantages[-1]
    grew = low[1] > high[1]
    all_positive = all(a > 0 for _, a in advantages)

    if all_positive and grew:
        out.append(
            f"VERDICT: hypothesis HELD. Modular wins at every budget, and the advantage "
            f"grows as data shrinks ({high[1]:+.5f} at frac={high[0]:g} -> "
            f"{low[1]:+.5f} at frac={low[0]:g})."
        )
    elif all_positive:
        out.append(
            f"VERDICT: PARTIAL. Modular wins at every budget, but the advantage does NOT "
            f"grow as data shrinks ({high[1]:+.5f} at frac={high[0]:g} -> "
            f"{low[1]:+.5f} at frac={low[0]:g}). The result is 'modular is better', "
            f"not 'modularity is a low-data mechanism' — a weaker and less novel claim."
        )
    else:
        out.append(
            "VERDICT: hypothesis REFUTED at one or more budgets. The modularity story "
            "does not hold as stated and the paper needs a different technical core."
        )
    return "\n".join(out)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("run_dirs", nargs="+")
    ap.add_argument("--out-dir", default="docs/figures")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    for rd in args.run_dirs:
        run_dir = Path(rd)
        try:
            data = load(run_dir)
        except FileNotFoundError as exc:
            print(f"skip {rd}: {exc}")
            continue

        source = data.get("split", {})
        title = (
            f"{run_dir.name}  —  {source.get('n_train', '?')} train / "
            f"{source.get('n_test_seen', '?')} seen / "
            f"{source.get('n_test_unseen', '?')} unseen trajectories, "
            f"{source.get('n_transitions_held_out', '?')} of "
            f"{source.get('n_transitions_total', '?')} compositions held out"
        )
        path = plot_run(data, title, out_dir / f"{run_dir.name}.png")
        print(f"\n{'=' * 78}\n{run_dir.name}\n{'=' * 78}")
        print(verdict(data))
        print(f"\nfigure -> {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
