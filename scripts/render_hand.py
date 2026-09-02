"""Render hand trajectories as animated GIFs.

The project's outputs so far are all statistics. This turns a 27-DOF trajectory
back into something you can look at, which serves two purposes:

* **Sanity.** Every convention in this repo -- the flexion axis, the flexion
  sign, the DIP/PIP coupling ratio, the limit box -- was inferred from data and
  verified only numerically. A wrong sign gives a hand that bends backwards, and
  no residual statistic makes that as obvious as watching it once. The OakInk
  adapter's first version had exactly that bug and pinned 41% of DOF values at
  their limits before anyone saw a picture.
* **Communication.** Primitive rollouts side by side are the one asset that
  shows what "a library of reusable bricks" means, and they need this renderer.

Draws the stick-figure FK from :mod:`caredex.kinematics`, not the MANO mesh:
the 27-DOF model is what every experiment consumes, so this shows exactly what
the models see. The mesh would look better and prove less.

    python scripts/render_hand.py --bundle data/bundles/grab.npz --label-contains drink
    python scripts/render_hand.py --bundle data/bundles/grab.npz --grid 6 --out docs/figures/grab_grid.gif
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.animation import FuncAnimation, PillowWriter  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from caredex.data.base import TrajectoryBundle  # noqa: E402
from caredex.kinematics import all_chains  # noqa: E402

DIGITS = ("thumb", "index", "middle", "ring", "pinky")
COLOURS = {
    "thumb": "#DC2626",
    "index": "#EA580C",
    "middle": "#CA8A04",
    "ring": "#059669",
    "pinky": "#2563EB",
}


def chains_over_time(traj: np.ndarray) -> dict[str, np.ndarray]:
    """``(T, 27)`` -> per-digit ``(T, 4, 3)`` joint positions."""
    return all_chains(np.asarray(traj, dtype=np.float64))


def axis_limits(all_pts: list[np.ndarray]) -> tuple[np.ndarray, float]:
    """Shared cube so every panel is at the same scale and hands are comparable."""
    pts = np.concatenate([p.reshape(-1, 3) for p in all_pts])
    centre = (pts.max(0) + pts.min(0)) / 2
    half = float((pts.max(0) - pts.min(0)).max()) / 2 * 1.15
    return centre, max(half, 1e-3)


def draw(ax, chains: dict[str, np.ndarray], t: int, centre: np.ndarray, half: float,
         title: str | None) -> None:
    ax.clear()
    for d in DIGITS:
        pts = np.concatenate([np.zeros((1, 3)), chains[d][t]])  # wrist -> tip
        ax.plot(pts[:, 0], pts[:, 1], pts[:, 2], "-o", color=COLOURS[d],
                lw=2.0, ms=3.4, mec="none")
    ax.scatter([0], [0], [0], color="#111827", s=26)
    ax.set_xlim(centre[0] - half, centre[0] + half)
    ax.set_ylim(centre[1] - half, centre[1] + half)
    ax.set_zlim(centre[2] - half, centre[2] + half)
    ax.set_axis_off()
    ax.view_init(elev=22, azim=-62)
    if title:
        ax.set_title(title, fontsize=8.5, pad=-6)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bundle", default="data/bundles/grab.npz")
    ap.add_argument("--out", default="docs/figures/hand.gif")
    ap.add_argument("--label-contains", default=None,
                    help="only trajectories whose label contains this substring")
    ap.add_argument("--grid", type=int, default=1, help="how many trajectories to tile")
    ap.add_argument("--max-frames", type=int, default=120)
    ap.add_argument("--fps", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    b = TrajectoryBundle.load(ROOT / args.bundle)
    idx = [i for i, l in enumerate(b.labels)
           if args.label_contains is None or args.label_contains in l]
    if not idx:
        print(f"no trajectory label contains {args.label_contains!r}")
        return 1

    rng = np.random.default_rng(args.seed)
    # One per distinct label where possible -- tiling six clips of the same
    # grasp shows variance, not vocabulary.
    seen, picks = set(), []
    for i in rng.permutation(idx):
        if b.labels[i] in seen:
            continue
        seen.add(b.labels[i])
        picks.append(int(i))
        if len(picks) == args.grid:
            break
    for i in rng.permutation(idx):  # top up if labels ran out
        if len(picks) == args.grid:
            break
        if int(i) not in picks:
            picks.append(int(i))

    clips = [chains_over_time(b.trajectories[i][: args.max_frames]) for i in picks]
    n_frames = min(min(c["index"].shape[0] for c in clips), args.max_frames)
    centre, half = axis_limits([np.stack([c[d] for d in DIGITS]) for c in clips])

    cols = int(np.ceil(np.sqrt(len(clips))))
    rows = int(np.ceil(len(clips) / cols))
    fig = plt.figure(figsize=(3.0 * cols, 3.0 * rows))
    axes = [fig.add_subplot(rows, cols, k + 1, projection="3d") for k in range(len(clips))]
    titles = [b.labels[i] for i in picks] if len(clips) > 1 else [b.labels[picks[0]]]

    def update(t: int):
        for ax, c, title in zip(axes, clips, titles):
            draw(ax, c, t, centre, half, title)
        return axes

    fig.suptitle(f"{Path(args.bundle).stem}  ({b.fps:g} fps)", fontsize=10)
    fig.tight_layout()
    anim = FuncAnimation(fig, update, frames=n_frames, interval=1000 / args.fps)

    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    anim.save(out, writer=PillowWriter(fps=args.fps))
    plt.close(fig)
    print(f"wrote {out}  ({len(clips)} clip(s), {n_frames} frames)")
    for i in picks:
        print(f"   {b.labels[i]:<28} {len(b.trajectories[i])} frames")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
