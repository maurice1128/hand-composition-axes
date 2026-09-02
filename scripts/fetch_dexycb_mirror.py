"""Fetch the DexYCB annotation-only mirror, surviving Hugging Face rate limits.

Unauthenticated requests to the Hub get throttled, and a plain
``snapshot_download`` of 7,838 small files dies with HTTP 429 partway through.
This retries with exponential backoff; the underlying call resumes from
whatever is already on disk, so an interrupted run costs only the backoff, not
the bytes.

Setting ``HF_TOKEN`` (any free read token) raises the limits considerably and
makes the retries unnecessary -- but the download works without one.

    python scripts/fetch_dexycb_mirror.py
    python scripts/fetch_dexycb_mirror.py --dest E:\\datasets\\dexycb_vitra
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

REPO = "MIT-Media-Lab/dexycb-vitra-streaming-v2"
EXPECTED_FILES = 7838


def count_npy(dest: Path) -> tuple[int, float]:
    files = list(dest.rglob("*.npy"))
    return len(files), sum(f.stat().st_size for f in files) / 1e6


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dest", default=r"E:\datasets\dexycb_vitra")
    ap.add_argument("--max-attempts", type=int, default=40)
    ap.add_argument("--base-delay", type=float, default=20.0)
    ap.add_argument("--max-delay", type=float, default=300.0)
    ap.add_argument("--workers", type=int, default=2)
    args = ap.parse_args()

    from huggingface_hub import snapshot_download

    dest = Path(args.dest)
    if not os.environ.get("HF_TOKEN"):
        print("[fetch] no HF_TOKEN set -- expect throttling and several retries\n")

    for attempt in range(1, args.max_attempts + 1):
        n, mb = count_npy(dest)
        print(f"[fetch] attempt {attempt}/{args.max_attempts}  have {n:,}/{EXPECTED_FILES:,} files ({mb:,.0f} MB)")
        try:
            snapshot_download(
                repo_id=REPO,
                repo_type="dataset",
                local_dir=str(dest),
                allow_patterns=["Annotation/**"],
                max_workers=args.workers,
            )
        except Exception as exc:  # noqa: BLE001 - any transport failure is retryable here
            n, mb = count_npy(dest)
            # Exponential backoff, capped. The cap matters: HF's limit resets on
            # a timescale of minutes, so waiting an hour buys nothing.
            delay = min(args.base_delay * (1.7 ** (attempt - 1)), args.max_delay)
            print(f"[fetch]   {type(exc).__name__}: {str(exc)[:160]}")
            print(f"[fetch]   now at {n:,} files ({mb:,.0f} MB); sleeping {delay:.0f}s\n")
            time.sleep(delay)
            continue

        n, mb = count_npy(dest)
        print(f"\n[fetch] complete: {n:,} files, {mb:,.0f} MB in {dest}")
        if n < EXPECTED_FILES:
            print(
                f"[fetch] WARNING: expected {EXPECTED_FILES:,} .npy files but found {n:,}. "
                "The mirror may have changed; check before treating the set as complete."
            )
        return 0

    n, mb = count_npy(dest)
    print(f"\n[fetch] gave up after {args.max_attempts} attempts with {n:,}/{EXPECTED_FILES:,} files.")
    print("[fetch] the partial set is usable -- the adapter reads whatever is present --")
    print("[fetch] but report the actual sequence count, not the dataset's nominal size.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
