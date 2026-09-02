"""Diagnose which MANO arrays the chumpy-free loader failed to unwrap.

``shapedirs`` and ``J_regressor`` came back empty on the first load. Both are
wrapped -- shapedirs in a chumpy array, J_regressor in a scipy sparse matrix --
and the stub unwrapper did not know their state layout. This prints the raw
state so the unwrapper can be fixed against what is actually stored rather than
against a guess.
"""

from __future__ import annotations

import io
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from caredex.mano import _Stub, _Unpickler  # noqa: E402

PATH = sys.argv[1] if len(sys.argv) > 1 else r"D:\datasets\mano\MANO_RIGHT.pkl"


def show(name: str, obj, depth: int = 0) -> None:
    pad = "  " * (depth + 1)
    if isinstance(obj, _Stub):
        state = getattr(obj, "_state", None)
        args = getattr(obj, "_args", ())
        print(f"{pad}{name}: _Stub  args={_brief(args)}")
        if isinstance(state, dict):
            print(f"{pad}  state keys: {sorted(state)}")
            for k, v in state.items():
                print(f"{pad}    {k}: {_brief(v)}")
        elif isinstance(state, tuple):
            print(f"{pad}  state tuple len={len(state)}")
            for i, v in enumerate(state):
                print(f"{pad}    [{i}]: {_brief(v)}")
        else:
            print(f"{pad}  state: {_brief(state)}")
    else:
        print(f"{pad}{name}: {_brief(obj)}")


def _brief(v) -> str:
    if isinstance(v, np.ndarray):
        return f"ndarray{v.shape} {v.dtype}"
    if isinstance(v, _Stub):
        return "_Stub"
    if isinstance(v, (tuple, list)):
        return f"{type(v).__name__}(len={len(v)}) [{', '.join(_brief(x) for x in v[:4])}]"
    if isinstance(v, dict):
        return f"dict{sorted(v)}"
    return f"{type(v).__name__} {v!r}"[:120]


def main() -> int:
    with open(PATH, "rb") as fh:
        raw = _Unpickler(io.BytesIO(fh.read()), encoding="latin1").load()
    print(f"{PATH}\ntop-level keys: {sorted(raw)}\n")
    for key in sorted(raw):
        show(key, raw[key])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
