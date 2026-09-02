"""Assemble the clips and title cards into a single explainer video.

The project page carries four clips, each of which needs a paragraph of context
before it means anything. A visitor who watches them in isolation sees four
hands moving. This puts them in order with the context on screen, so the whole
argument runs in about seventy seconds without anyone reading first.

Silent by design: there is no narration track, so every claim is carried as a
title card and the pacing is set by reading time rather than by speech. Cards
are held long enough to read twice at a slow pace, which is why the timings
below look generous.

The cards restate the paper's claims at the paper's own strength -- "descriptive"
where the paper says descriptive, "null" where it says null. A video is the part
of a project page people actually watch, so overstating here would undo the
care taken everywhere else.

    python scripts/build_explainer_video.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]

W, H, FPS = 1280, 720, 25

#: Matches the project page's light palette so the video does not look like a
#: separate artefact bolted on.
PAPER = (252, 251, 249)
INK = (22, 25, 26)
INK2 = (65, 71, 74)
INK3 = (108, 115, 119)
RUST = (158, 54, 32)
BLUE = (43, 100, 128)
GREEN = (47, 107, 60)

FONTS = {
    "serif": r"C:\Windows\Fonts\georgia.ttf",
    "serif_b": r"C:\Windows\Fonts\georgiab.ttf",
    "sans": r"C:\Windows\Fonts\segoeui.ttf",
    "mono": r"C:\Windows\Fonts\consola.ttf",
}


def font(name: str, size: int):
    from PIL import ImageFont

    try:
        return ImageFont.truetype(FONTS[name], size)
    except OSError:
        return ImageFont.load_default()


def wrap(draw, text: str, f, width: int) -> list[str]:
    words, lines, cur = text.split(), [], ""
    for w in words:
        trial = f"{cur} {w}".strip()
        if draw.textlength(trial, font=f) <= width:
            cur = trial
        else:
            if cur:
                lines.append(cur)
            cur = w
    if cur:
        lines.append(cur)
    return lines


def card(eyebrow: str, headline: str, body: str = "", accent=RUST,
         figures: list[tuple[str, str]] | None = None):
    """One title card as a single RGB frame."""
    from PIL import Image, ImageDraw

    img = Image.new("RGB", (W, H), PAPER)
    d = ImageDraw.Draw(img)
    margin = 128
    inner = W - 2 * margin

    f_eye = font("mono", 21)
    f_head = font("serif", 56)
    f_body = font("sans", 27)
    f_k = font("mono", 19)
    f_v = font("serif", 40)

    # Vertical layout is computed first so the block sits optically centred
    # rather than pinned to the top -- a card that drifts upward reads as a
    # slide someone forgot to finish.
    head_lines = wrap(d, headline, f_head, inner)
    body_lines = wrap(d, body, f_body, int(inner * 0.86)) if body else []
    h = 34 + len(head_lines) * 70
    if body_lines:
        h += 26 + len(body_lines) * 42
    if figures:
        h += 52
    y = (H - h) // 2

    d.text((margin, y), eyebrow.upper(), font=f_eye, fill=INK3)
    d.line([(margin, y + 34), (margin + 46, y + 34)], fill=accent, width=2)
    y += 60

    for line in head_lines:
        d.text((margin, y), line, font=f_head, fill=INK)
        y += 70
    y += 12

    for line in body_lines:
        d.text((margin, y), line, font=f_body, fill=INK2)
        y += 42

    if figures:
        y += 22
        x = margin
        for k, v in figures:
            d.text((x, y), k.upper(), font=f_k, fill=INK3)
            d.text((x, y + 26), v, font=f_v, fill=accent)
            x += max(int(d.textlength(v, font=f_v)),
                     int(d.textlength(k.upper(), font=f_k))) + 74
    return np.asarray(img)


def clip(path: Path, seconds: float, caption: str = "") -> list[np.ndarray]:
    """A GIF letterboxed onto the canvas, looped or trimmed to `seconds`."""
    import imageio.v3 as iio
    from PIL import Image, ImageDraw

    frames = iio.imread(path, index=None)
    want = int(seconds * FPS)
    idx = [i % len(frames) for i in range(want)]

    band = 76 if caption else 0
    box_w, box_h = W - 96, H - 96 - band
    out = []
    f_cap = font("sans", 23)

    for i in idx:
        src = Image.fromarray(frames[i][..., :3])
        scale = min(box_w / src.width, box_h / src.height)
        src = src.resize((int(src.width * scale), int(src.height * scale)),
                         Image.LANCZOS)
        canvas = Image.new("RGB", (W, H), PAPER)
        canvas.paste(src, ((W - src.width) // 2, (H - band - src.height) // 2))
        if caption:
            d = ImageDraw.Draw(canvas)
            lines = wrap(d, caption, f_cap, W - 200)
            ty = H - band + 12 - (len(lines) - 1) * 15
            for line in lines:
                d.text(((W - d.textlength(line, font=f_cap)) / 2, ty),
                       line, font=f_cap, fill=INK3)
                ty += 30
        out.append(np.asarray(canvas))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="docs/figures/explainer.mp4")
    ap.add_argument("--quality", type=int, default=8)
    args = ap.parse_args()

    import imageio

    R = ROOT / "runs"
    seq: list[tuple[str, object]] = [
        ("card", card("Measurement study",
                      "When do learned motion primitives compose?",
                      "Modular motion priors assume that primitives learned separately can be "
                      "recombined. The whole family rests on it. Nobody had measured it.")),
        ("hold", 4.0),

        ("card", card("The bank", "Sixteen primitives, discovered not labelled",
                      "No segmentation labels. The model finds its own decomposition from "
                      "reconstruction, and a consistency term asks each primitive to move the "
                      "same way whatever preceded it.", accent=BLUE)),
        ("hold", 4.0),
        ("clip", (R / "primitive_library.gif", 7.0,
                  "Every primitive decoded from the same starting pose. Bottom row composes two.")),

        ("card", card("The composition space", "168 of the 240 transitions have no precedent",
                      "Sixteen primitives admit 240 transitions between them. Training data "
                      "contains 72. Every composition test draws from the other 168.",
                      figures=[("seen", "72"), ("never seen", "168"), ("possible", "240")])),
        ("hold", 5.0),
        ("clip", (R / "composition_seen_vs_unseen.gif", 8.0,
                  "Left: a transition present in training. Right: one absent from it. "
                  "Both end on the same primitive.")),

        ("card", card("Is it real motion?", "Joint limits prove nothing here",
                      "Limits hold by construction in every arm. The check with teeth is "
                      "coupling: the distal joints of a real finger move together.",
                      accent=BLUE)),
        ("hold", 4.0),
        ("clip", (R / "validity_random_vs_generated.gif", 9.0,
                  "Uniform noise 31.0 deg of coupling error. Generated 8.9. Real human motion 8.9.")),

        ("card", card("Onto the robot", "Retargeted to a Shadow Hand",
                      "An object rests in the palm, the hand executes the motion as position "
                      "targets, then gravity swings sideways. Shown: the scripted-fist control "
                      "that has to hold before anything generated counts.", accent=BLUE)),
        ("hold", 4.0),
        ("clip", (R / "sample_grasp.gif", 6.0, "")),

        ("card", card("What we found", "Three findings, at three strengths",
                      "Modularity reduces the compositional penalty where the data has "
                      "interaction, confirmed. Where that happens is predictable from raw data, "
                      "descriptive. On a robot hand, unseen and seen are not distinguishable "
                      "-- a null, not an equivalence.",
                      figures=[("confirmed", "p = 0.0012"), ("descriptive", "r = 0.863"),
                               ("null result", "18 of 19")], accent=GREEN)),
        ("hold", 7.0),

        ("card", card("And what it does not show",
                      "One dataset of four carries the effect",
                      "The effect is small per split. The screen rests on six points, three "
                      "from one dataset. Transfer is kinematic and measures no force. Whether "
                      "modularity reaches a task is open.")),
        ("hold", 6.0),
    ]

    frames: list[np.ndarray] = []
    pending: np.ndarray | None = None
    for kind, payload in seq:
        if kind == "card":
            pending = payload
        elif kind == "hold":
            assert pending is not None
            frames += [pending] * int(payload * FPS)
            pending = None
        else:
            path, secs, cap = payload
            if not path.exists():
                print(f"missing clip, skipping: {path.name}")
                continue
            frames += clip(path, secs, cap)

    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimwrite(out, frames, fps=FPS, quality=args.quality,
                     macro_block_size=8)
    print(f"wrote {out}")
    print(f"  {len(frames)} frames at {FPS} fps = {len(frames) / FPS:.1f} s")
    print(f"  {out.stat().st_size / 1024 / 1024:.2f} MB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
