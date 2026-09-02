"""The primitive library, and a composition it was never trained on.

This is the figure that shows what the modular prior actually learned, and the
only asset in the project that lets a reader see the claim rather than read a
p-value. Two panels, made from the same model:

**The library.** Every primitive decoded on its own from the same neutral start,
so differences between panels are the primitive and nothing else. If the bank
were a context-dependent code the panels would look interchangeable; the
identity ratio says they should not.

**A composition.** Two primitives concatenated into a sequence the model was
never trained on, decoded in one pass. `demo_composition.py` already shows
numerically that unseen pairs decode no worse than seen ones -- this is the same
fact in a form that can go in a talk.

Rendered from the stick-figure FK, not the MANO mesh, for the same reason
`render_hand.py` is: the 27-DOF vector is what every experiment consumes, and a
mesh would look better while proving less.

    python scripts/figure_primitive_library.py --run-dir runs/prior_oakink2_consist
    python scripts/figure_primitive_library.py --run-dir runs/prior_oakink2_consist --gif
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from matplotlib.animation import FuncAnimation, PillowWriter  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from caredex.hand_model import denormalize  # noqa: E402
from render_hand import COLOURS, DIGITS, axis_limits, chains_over_time, draw  # noqa: E402

from demo_composition import load_model  # noqa: E402


def rollout(model, sequence: list[int], device, seed: int = 0) -> np.ndarray:
    """Decode a primitive sequence to native-unit DOF, ``(T, 27)``."""
    torch.manual_seed(seed)
    with torch.no_grad():
        w = model.sample(1, temperature=0.0, device=device, sequence=sequence)
    x = w[0].cpu().numpy() if torch.is_tensor(w) else np.asarray(w)[0]
    return denormalize(x[:, : 27])


def fingertip_trace(ax, chains, centre, half, upto: int) -> None:
    """Draw where each fingertip has travelled so far.

    Key frames alone made the panels look identical: the primitives differ by
    28.6 degrees of mean pose against a 15.4 degree within-rollout amplitude,
    which is a real difference but a small one to read off five static hands.
    The trace makes the motion itself visible, which is what a primitive *is*.
    """
    for d in DIGITS:
        tip = chains[d][: upto + 1, 3, :]
        if len(tip) > 1:
            ax.plot(tip[:, 0], tip[:, 1], tip[:, 2], "-", color=COLOURS[d],
                    lw=1.0, alpha=0.45)


def strip(ax_row, chains, centre, half, title, n_frames, colour_title="#111827"):
    """Draw one motion as a row of key frames, each with the path so far."""
    idx = np.linspace(0, n_frames - 1, len(ax_row)).astype(int)
    for k, (ax, t) in enumerate(zip(ax_row, idx)):
        draw(ax, chains, int(t), centre, half, None)
        fingertip_trace(ax, chains, centre, half, int(t))
        if k == 0:
            ax.text2D(-0.05, 0.88, title, transform=ax.transAxes,
                      fontsize=10, color=colour_title, weight="bold")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-dir", default="runs/prior_oakink2_consist")
    ap.add_argument("--bundle", default="data/bundles/oakink2.npz")
    ap.add_argument("--out", default="docs/figures/primitive_library.png")
    ap.add_argument("--n-show", type=int, default=4, help="primitives to draw")
    ap.add_argument("--keyframes", type=int, default=5)
    ap.add_argument("--gif", action="store_true", help="also write an animated version")
    ap.add_argument("--compose", type=int, nargs=2, default=None,
                    help="which two primitives to concatenate (default: the two "
                         "whose solo rollouts differ most)")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, _ = load_model(Path(args.run_dir), "best.pt", device)
    k = model.cfg.n_primitives

    solo = {p: rollout(model, [p], device) for p in range(k)}

    # Show the most distinct primitives rather than the first N: with 16 in the
    # bank and room for 8, picking by index would be picking arbitrarily.
    means = np.stack([solo[p].mean(axis=0) for p in range(k)])
    spread = np.linalg.norm(means - means.mean(axis=0), axis=1)
    shown = sorted(np.argsort(-spread)[: args.n_show].tolist())

    if args.compose is None:
        d = [(np.linalg.norm(means[a] - means[b]), a, b)
             for i, a in enumerate(shown) for b in shown[i + 1:]]
        _, a, b = max(d)
    else:
        a, b = args.compose
    pair = rollout(model, [a, b], device)

    clips = [(f"primitive {p}", chains_over_time(solo[p])) for p in shown]
    clips.append((f"primitive {a} then {b}  (never trained as a pair)",
                  chains_over_time(pair)))
    centre, half = axis_limits([np.stack([c[d_] for d_ in DIGITS]) for _, c in clips])
    half *= 0.80
    n_frames = min(c["index"].shape[0] for _, c in clips)

    rows, cols = len(clips), args.keyframes
    fig = plt.figure(figsize=(2.3 * cols, 2.0 * rows))
    axes = [[fig.add_subplot(rows, cols, r * cols + c + 1, projection="3d")
             for c in range(cols)] for r in range(rows)]
    for r, (title, chains) in enumerate(clips):
        strip(axes[r], chains, centre, half, title,
              n_frames, "#B91C1C" if r == len(clips) - 1 else "#111827")
    fig.suptitle(
        f"{Path(args.run_dir).name}: {k} learned primitives, decoded from one start pose\n"
        "thumb=red  index=orange  middle=yellow  ring=green  pinky=blue",
        fontsize=9.5)
    # 3D axes reserve a large margin for decorations that set_axis_off does not
    # reclaim, so tight_layout leaves most of the page blank. Overlap them.
    for row in axes:
        for ax in row:
            ax.set_box_aspect((1, 1, 1), zoom=1.22)
    fig.subplots_adjust(left=0.005, right=0.995, top=0.93, bottom=0.005,
                        wspace=-0.35, hspace=-0.30)

    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=165)
    plt.close(fig)
    print(f"wrote {out}")
    print(f"  primitives shown: {shown}")
    print(f"  composition: {a} -> {b}, {len(pair)} frames")

    if args.gif:
        cols_g = min(4, len(clips))
        rows_g = int(np.ceil(len(clips) / cols_g))
        figg = plt.figure(figsize=(2.7 * cols_g, 2.7 * rows_g))
        gaxes = [figg.add_subplot(rows_g, cols_g, i + 1, projection="3d")
                 for i in range(len(clips))]

        def update(t: int):
            for ax, (title, chains) in zip(gaxes, clips):
                draw(ax, chains, t, centre, half, title)
            return gaxes

        anim = FuncAnimation(figg, update, frames=n_frames, interval=50)
        gif = out.with_suffix(".gif")
        anim.save(gif, writer=PillowWriter(fps=20))
        plt.close(figg)
        print(f"wrote {gif}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
