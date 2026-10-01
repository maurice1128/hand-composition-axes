"""Paper figures from the two project-page videos, with the burned-in text bars cropped off and
labels set in the paper's font at print size.

cube_frames: rows = keyframe / continuous latent, columns = 1.5 s, 3.0 s, 5.0 s of the held-out
episode in runs/rl/video/motion_level.mp4 (hold y, hold x, end); the latent panel freezes after it
loses the cube at 1.8 s, as in the video.
arms_frames: rows = during the grasp / end, columns = A, B, C from runs/lineC/video/arms.mp4.

    python scripts/figure_video_frames.py
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
TMP = ROOT / "runs" / "video_frames"
OUT = ROOT / "docs" / "submission" / "figures"
FF = "ffmpeg"


def grab(video, out, t=None, n=None):
    if t is not None:
        cmd = [FF, "-loglevel", "error", "-y", "-ss", str(t), "-i", str(video), "-frames:v", "1", str(out)]
    else:
        cmd = [FF, "-loglevel", "error", "-y", "-i", str(video), "-vf", f"select=eq(n\\,{n})",
               "-frames:v", "1", "-fps_mode", "passthrough", str(out)]
    subprocess.run(cmd, check=True)
    return np.asarray(Image.open(out).convert("RGB"))


def n_frames(video):
    r = subprocess.run(["ffprobe", "-v", "error", "-count_frames", "-select_streams", "v", "-show_entries",
                        "stream=nb_read_frames", "-of", "csv=p=0", str(video)], capture_output=True, text=True)
    return int(r.stdout.strip())


def sheet(panels, rows, cols, col_titles, row_titles, notes, out, width=6.4):
    plt.rcParams.update({"font.size": 8, "font.family": "serif"})
    h, w = panels[0][0].shape[:2]
    fig, axes = plt.subplots(rows, cols, figsize=(width, width * rows * h / (cols * w) + 0.35))
    for r in range(rows):
        for c in range(cols):
            ax = axes[r, c]
            ax.imshow(panels[r][c])
            ax.set_xticks([]); ax.set_yticks([])
            for s in ax.spines.values():
                s.set_visible(False)
            if r == 0:
                ax.set_title(col_titles[c], fontsize=8, pad=3)
            if c == 0:
                ax.set_ylabel(row_titles[r], fontsize=8)
            note = notes[r][c]
            if note:
                text, color = note
                ax.text(0.03, 0.05, text, transform=ax.transAxes, fontsize=7.5, color="white",
                        bbox=dict(boxstyle="square,pad=0.25", fc=color, ec="none"))
    fig.subplots_adjust(left=0.05, right=0.995, top=0.93, bottom=0.01, wspace=0.03, hspace=0.05)
    fig.savefig(out, dpi=300)
    plt.close(fig)


def main() -> int:
    TMP.mkdir(parents=True, exist_ok=True)
    red, green = "#9E3431", "#23613F"

    # cube: panels are 480 wide with an 8 px gap; crop off the text bars and zoom on the hand
    cube = ROOT / "runs" / "rl" / "video" / "motion_level.mp4"
    frames = [grab(cube, TMP / f"cube_{t}.png", t=t) for t in (1.5, 3.0, 5.0)]
    crop = lambda f, x0: f[40:300, x0 + 25:x0 + 345]
    panels = [[crop(f, 0) for f in frames], [crop(f, 488) for f in frames]]
    notes = [[None, None, ("succeeded", green)],
             [None, ("cube lost at 1.8 s", red), ("cube lost at 1.8 s", red)]]
    sheet(panels, 2, 3, ["1.5 s: hold, tilt $y$", "3.0 s: hold, tilt $x$", "5.0 s: end"],
          ["keyframe", "continuous latent"], notes, OUT / "cube_frames.png")

    # arms: three 480-wide panels; crop to the table top
    arms = ROOT / "runs" / "lineC" / "video" / "arms.mp4"
    n = n_frames(arms)
    start, end = grab(arms, TMP / "arms_start.png", n=60), grab(arms, TMP / "arms_end.png", n=n - 5)
    crop = lambda f, i: f[70:330, i * 480 + 70:i * 480 + 410]
    panels = [[crop(start, i) for i in range(3)], [crop(end, i) for i in range(3)]]
    notes = [[None, None, None],
             [("1 of 3 stages", red), ("all 3 stages", green), ("0 of 3 stages", red)]]
    sheet(panels, 2, 3, ["A: best flat policy", "B: best flat policy per stage", "C: separate skills"],
          ["during the grasp", "end"], notes, OUT / "arms_frames.png")
    print("wrote", OUT / "cube_frames.png", OUT / "arms_frames.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
