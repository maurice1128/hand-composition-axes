"""A short explainer video for the project page: why a held-out label combination only holds out the motion when the
label describes the scored motion. Schematic animation; every number on screen is from the manuscript
(docs/tmlr/PAPER_TMLR.md, checked by scripts/verify_tmlr_draft.py):

    aligned two-clip OakInk-Image trajectories   +17.0 % of the naive error   (Table 3, 40 seeds)
    misaligned two-clip trajectories              +3.9 %                       (Table 3, 40 seeds)
    zero-truth control at the same budget         +3.0 %                       (Table 3 caption, 20 seeds)
    TACO, registered in advance                   +10.6 %                      (Section 4.4, 40 seeds)

Usage: python scripts/make_web_video_label_alignment.py OUT_DIR
Writes label_alignment.mp4 (H.264, 30 fps, 1280x720) and label_alignment_poster.png.
"""
import subprocess
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import FancyBboxPatch, Rectangle  # noqa: E402

FPS, W, H = 30, 12.8, 7.2
FFMPEG = r"C:\Users\maurice\AppData\Local\Microsoft\WinGet\Packages\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe\ffmpeg-9.0.1-full_build\bin\ffmpeg.exe"

RED, BLUE, GREEN, GOLD = "#d62728", "#1f77b4", "#2ca02c", "#e6a100"
INK, MUTED, PANEL = "#1a1a1a", "#6b6b6b", "#f3f3f3"
AL = {"aligned": 17.03, "misaligned": 3.87, "control": 2.96, "taco": 10.63}

# Six recordings: (label colour, content of first half, content of second half) for each scene.
REC_ALIGNED = [(RED, RED, RED), (BLUE, BLUE, BLUE), (GREEN, GREEN, GREEN),
               (RED, RED, RED), (BLUE, BLUE, BLUE), (GREEN, GREEN, GREEN)]
REC_MISALIGNED = [(RED, RED, BLUE), (BLUE, BLUE, RED), (GREEN, GREEN, BLUE),
                  (RED, RED, GREEN), (BLUE, BLUE, GREEN), (GREEN, GREEN, RED)]


def ease(t):
    t = np.clip(t, 0, 1)
    return t * t * (3 - 2 * t)


def seg(t, a, b):
    return ease((t - a) / (b - a))


def draw_scene(ax, t, recs, title, sub, value, note, leak):
    """One 8-second scene. t in seconds from the scene start."""
    ax.text(0.5, 0.945, title, ha="center", va="center", fontsize=25, weight="bold", color=INK,
            alpha=seg(t, 0, 0.6))
    ax.text(0.5, 0.885, sub, ha="center", va="center", fontsize=15, color=MUTED, alpha=seg(t, 0.2, 0.8))
    # Boxes: held out (top right), training set (bottom right).
    for (x, y, w, h, lab, col) in ((0.47, 0.60, 0.24, 0.20, "held out", RED), (0.47, 0.13, 0.24, 0.40,
                                                                                  "training set", INK)):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.008,rounding_size=0.015",
                                    fc=PANEL, ec=col, lw=1.6, alpha=seg(t, 0.5, 1.0)))
        ax.text(x + 0.012, y + h - 0.03, lab, fontsize=13, color=col, weight="bold", alpha=seg(t, 0.5, 1.0))
    ax.text(0.03, 0.80, "recordings, each with one label", fontsize=13, color=MUTED, alpha=seg(t, 0.3, 0.9))
    move = seg(t, 2.0, 3.6)
    held_slot, train_slot = 0, 0
    for i, (lab, c1, c2) in enumerate(recs):
        x0, y0 = 0.05, 0.73 - i * 0.095
        if lab == RED:
            x1, y1 = 0.50, 0.70 - held_slot * 0.075
            held_slot += 1
        else:
            x1, y1 = 0.50, 0.43 - train_slot * 0.07
            train_slot += 1
        x, y = x0 + (x1 - x0) * move, y0 + (y1 - y0) * move
        a = seg(t, 0.6 + 0.12 * i, 1.2 + 0.12 * i)
        ax.add_patch(Rectangle((x + 0.035, y), 0.075, 0.045, fc=c1, ec="white", lw=1.5, alpha=a))
        ax.add_patch(Rectangle((x + 0.11, y), 0.075, 0.045, fc=c2, ec="white", lw=1.5, alpha=a))
        ax.add_patch(Rectangle((x, y + 0.008), 0.028, 0.029, fc=lab, ec=INK, lw=0.8, alpha=a))
        ax.text(x + 0.014, y + 0.0225, "L", ha="center", va="center", fontsize=9, color="white", weight="bold",
                alpha=a)
        if leak and lab != RED and c2 == RED and t > 3.9:
            blink = 0.5 + 0.5 * np.sin((t - 3.9) * 7)
            ax.add_patch(Rectangle((x + 0.107, y - 0.006), 0.081, 0.057, fc="none", ec=GOLD, lw=3.2, alpha=blink))
    if leak:
        ax.text(0.59, 0.075, "held-out motion slips in under another label", ha="center", fontsize=13,
                color=GOLD, weight="bold", alpha=seg(t, 4.0, 4.6))
    else:
        ax.text(0.59, 0.075, "no held-out motion in training", ha="center", fontsize=13, color=GREEN,
                weight="bold", alpha=seg(t, 4.0, 4.6))
    # Legend: label chip vs content.
    ax.text(0.03, 0.12, "L = the label        coloured blocks = the motion in each half", fontsize=11,
            color=MUTED, alpha=seg(t, 0.8, 1.4))
    # Meter.
    mx, my, mw, mh = 0.80, 0.18, 0.08, 0.62
    top = 20.0
    ax.add_patch(Rectangle((mx, my), mw, mh, fc=PANEL, ec=INK, lw=1.2, alpha=seg(t, 0.5, 1.0)))
    grow = seg(t, 4.4, 6.0)
    ax.add_patch(Rectangle((mx, my), mw, mh * value / top * grow, fc=RED if value > 8 else MUTED, ec="none"))
    yc = my + mh * AL["control"] / top
    ax.plot([mx - 0.012, mx + mw + 0.012], [yc, yc], ls="--", color=INK, lw=1.4, alpha=seg(t, 0.8, 1.4))
    ax.text(mx - 0.018, yc, f"control +{AL['control']:.1f} %", va="center", ha="right", fontsize=11, color=INK,
            alpha=seg(t, 0.8, 1.4))
    ax.text(mx + mw / 2, my + mh + 0.035, "penalty", ha="center", fontsize=14, weight="bold", color=INK,
            alpha=seg(t, 0.5, 1.0))
    ax.text(mx + mw / 2, my - 0.045, "% of naive error", ha="center", fontsize=10, color=MUTED,
            alpha=seg(t, 0.5, 1.0))
    ax.text(mx + mw + 0.018, my + mh * value / top * grow, f"+{value * grow:.1f} %", va="center", fontsize=20,
            weight="bold", color=RED if value > 8 else INK, alpha=seg(t, 4.4, 4.8))
    ax.text(0.5, 0.02, note, ha="center", fontsize=12, color=MUTED, alpha=seg(t, 5.5, 6.2))


def draw_end(ax, t):
    ax.text(0.5, 0.66, "A hand-motion dataset can test compositional generalisation", ha="center",
            fontsize=21, weight="bold", color=INK, alpha=seg(t, 0, 0.8))
    ax.text(0.5, 0.58, "only when each label describes the whole trajectory it is attached to.", ha="center", fontsize=21,
            weight="bold", color=INK, alpha=seg(t, 0.2, 1.0))
    ax.text(0.5, 0.44, "OakInk-Image, same data volume:   label describes both halves  +17.0 %"
            "     only the first half  +3.9 %", ha="center", fontsize=14, color=INK, alpha=seg(t, 1.2, 1.8))
    ax.text(0.5, 0.37, f"TACO, one action per label, predicted before its data were opened:  "
            f"+{AL['taco']:.1f} %", ha="center", fontsize=14, color=INK, alpha=seg(t, 1.8, 2.4))
    ax.text(0.5, 0.22, "Hold out the motion, not just the label.", ha="center", fontsize=20, color=RED,
            weight="bold", alpha=seg(t, 2.6, 3.2))


SCENES = [
    (8.0, lambda ax, t: draw_scene(ax, t, REC_ALIGNED, "1  The label describes the whole recording",
                                   "hold out the red combination: every red recording leaves training",
                                   AL["aligned"], "OakInk-Image, two clips per recording, 40 seeds", False)),
    (8.5, lambda ax, t: draw_scene(ax, t, REC_MISALIGNED, "2  The label describes only the first half",
                                   "the same rule, the same amount of data", AL["misaligned"],
                                   "not distinguishable from the control", True)),
    (6.0, draw_end),
]

def main():
    OUT = Path(sys.argv[1] if len(sys.argv) > 1 else ".")
    OUT.mkdir(parents=True, exist_ok=True)
    frames = OUT / "_frames"
    frames.mkdir(exist_ok=True)
    for f in frames.glob("*.png"):
        f.unlink()
    k = 0
    for dur, fn in SCENES:
        for j in range(int(dur * FPS)):
            fig = plt.figure(figsize=(W, H), dpi=100)
            ax = fig.add_axes([0, 0, 1, 1])
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.axis("off")
            fig.patch.set_facecolor("white")
            t = j / FPS
            fn(ax, t)
            fade = min(1.0, t / 0.3, (dur - t) / 0.3)
            if fade < 1:
                ax.add_patch(Rectangle((0, 0), 1, 1, fc="white", ec="none", alpha=1 - max(fade, 0)))
            fig.savefig(frames / f"f{k:05d}.png", facecolor="white")
            if dur == 6.0 and j == int(4.5 * FPS):
                fig.savefig(OUT / "label_alignment_poster.png", facecolor="white")
            plt.close(fig)
            k += 1
    mp4 = OUT / "label_alignment.mp4"
    subprocess.run([FFMPEG, "-y", "-loglevel", "error", "-framerate", str(FPS), "-i", str(frames / "f%05d.png"),
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "26", "-preset", "slow", "-movflags", "+faststart",
                    str(mp4)], check=True)
    for f in frames.glob("*.png"):
        f.unlink()
    frames.rmdir()
    print(mp4, f"{mp4.stat().st_size / 1e6:.2f} MB", k, "frames")


if __name__ == "__main__":
    main()
