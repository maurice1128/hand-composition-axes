"""Generate (or load) a trajectory bundle and cache it to disk.

    python scripts/generate_synthetic.py
    python scripts/generate_synthetic.py --set data.synthetic.n_trajectories=2048
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from caredex.config import dump, get, load_config  # noqa: E402
from caredex.data import get_source  # noqa: E402
from caredex.data.base import TrajectoryBundle  # noqa: E402
from caredex.hand_model import summary  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=None)
    ap.add_argument("--set", dest="overrides", nargs="*", default=[])
    ap.add_argument("--out", default=None, help="override data.cache")
    ap.add_argument("--force", action="store_true", help="regenerate even if cached")
    ap.add_argument("--show-dofs", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.config, args.overrides)
    if args.show_dofs:
        print(summary())
        print()

    source_name = get(cfg, "data.source", "synthetic")
    out = Path(args.out or get(cfg, "data.cache", "data/bundles/bundle.npz"))

    if out.exists() and not args.force:
        bundle = TrajectoryBundle.load(out)
        print(f"[data] loaded cached bundle from {out} (use --force to regenerate)")
    else:
        if source_name == "synthetic":
            s = get(cfg, "data.synthetic", {})
            source = get_source(
                "synthetic",
                n_trajectories=s.get("n_trajectories", 512),
                fps=get(cfg, "data.fps", 30.0),
                seg_frames=tuple(s.get("seg_frames", (25, 60))),
                segments_per_traj=tuple(s.get("segments_per_traj", (3, 7))),
                noise_rank=s.get("noise_rank", 6),
                noise_deg=s.get("noise_deg", 7.0),
                sensor_noise_deg=s.get("sensor_noise_deg", 0.4),
                transition_via_deg=s.get("transition_via_deg", 0.0),
                seed=s.get("seed", 0),
            )
        else:
            source = get_source(source_name, root=get(cfg, "data.root"))
        bundle = source.load()
        bundle.save(out)
        print(f"[data] wrote {out}")

    print()
    print(bundle.describe())
    print()
    print(f"config:\n{dump({'data': cfg.get('data', {})})}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
