"""Load the MANO hand model without chumpy.

``MANO_RIGHT.pkl`` is a Python 2 pickle whose arrays are wrapped in ``chumpy``
autodiff objects. chumpy is unmaintained and does not import on modern numpy
(it references ``np.bool``, ``np.float`` and friends, removed in numpy 1.24).
Installing it means pinning numpy backwards for the whole project, which is not
worth it when the only thing needed from the file is a handful of plain arrays.

So the pickle is opened with a custom unpickler that substitutes a stub for
every chumpy class and then unwraps the stubs into numpy arrays.

What the model is needed for
----------------------------
Phase 2's ergonomic validation checks self-intersection. Until now that has
been a capsule-model proxy (:mod:`caredex.kinematics`) because the mesh was
behind a registration wall. With the mesh available, the check can move to
actual triangle geometry. The proxy stays as the fast path.

    from caredex.mano import load_mano
    m = load_mano(r"D:\\datasets\\mano\\MANO_RIGHT.pkl")
    m.faces           # (1538, 3) triangle indices
    m.v_template      # (778, 3) rest-pose vertices
    m.shapedirs       # (778, 3, 10) shape blend shapes
    m.posedirs        # (778, 3, 135) pose blend shapes
    m.J_regressor     # (16, 778) joint regressor
    m.weights         # (778, 16) linear blend skinning weights
"""

from __future__ import annotations

import io
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


class _Stub:
    """Stand-in for any class the pickle references that we cannot import.

    Chumpy arrays store their data in an ``x`` attribute; scipy sparse matrices
    reconstruct through ``__setstate__``. Accepting anything and remembering the
    state is enough to recover the arrays.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self._args = args

    def __setstate__(self, state: Any) -> None:
        self._state = state

    def __reduce__(self):  # pragma: no cover - never re-pickled
        raise NotImplementedError


class _Unpickler(pickle.Unpickler):
    def find_class(self, module: str, name: str):
        if module.startswith("chumpy") or module.startswith("scipy.sparse"):
            return _Stub
        try:
            return super().find_class(module, name)
        except (ImportError, AttributeError):
            return _Stub


def _unwrap(obj: Any) -> Any:
    """Turn stubs and chumpy wrappers into numpy arrays, recursively.

    Three wrapper shapes appear in MANO_RIGHT.pkl, all handled here:

    * plain chumpy ``Ch`` -- the array sits under ``x``/``_x``/``r``;
    * chumpy reorder nodes (``shapedirs``) -- an underlying array ``a`` plus an
      index vector ``idxs`` and a ``preferred_shape`` to gather and reshape by;
    * scipy sparse CSC (``J_regressor``) -- ``data``/``indices``/``indptr``
      plus ``_shape``, densified because it is only 16x778.
    """
    if isinstance(obj, _Stub):
        state = getattr(obj, "_state", None)

        if isinstance(state, dict):
            for key in ("x", "_x", "r"):
                if key in state:
                    return _unwrap(state[key])
            if {"data", "indices", "indptr", "_shape"} <= set(state):
                return _sparse_to_dense(state)
            if "idxs" in state and "a" in state:
                return _reorder(state)
            return {k: _unwrap(v) for k, v in state.items()}

        if state is not None:
            return _unwrap(state)
        return tuple(_unwrap(a) for a in getattr(obj, "_args", ()))

    if isinstance(obj, dict):
        return {k: _unwrap(v) for k, v in obj.items()}
    return obj


def _sparse_to_dense(state: dict) -> np.ndarray:
    """Rebuild a CSC (or CSR) sparse matrix as a dense array."""
    data = np.asarray(_unwrap(state["data"]))
    indices = np.asarray(_unwrap(state["indices"]))
    indptr = np.asarray(_unwrap(state["indptr"]))
    shape = tuple(int(v) for v in _unwrap(state["_shape"]))

    dense = np.zeros(shape, dtype=data.dtype)
    # CSC has one indptr entry per column plus one; CSR per row.
    if len(indptr) == shape[1] + 1:
        for col in range(shape[1]):
            lo, hi = indptr[col], indptr[col + 1]
            dense[indices[lo:hi], col] = data[lo:hi]
    elif len(indptr) == shape[0] + 1:
        for row in range(shape[0]):
            lo, hi = indptr[row], indptr[row + 1]
            dense[row, indices[lo:hi]] = data[lo:hi]
    else:
        raise ValueError(
            f"indptr length {len(indptr)} matches neither dimension of {shape}"
        )
    return dense


def _reorder(state: dict) -> np.ndarray:
    """Rebuild a chumpy reorder node: gather ``a`` by ``idxs``, then reshape."""
    base = np.asarray(_unwrap(state["a"])).ravel()
    idxs = np.asarray(_unwrap(state["idxs"]), dtype=np.int64)
    gathered = base[idxs]
    shape = state.get("preferred_shape")
    if shape is not None:
        gathered = gathered.reshape(tuple(int(v) for v in _unwrap(shape)))
    return gathered


@dataclass
class ManoModel:
    """The plain-array subset of the MANO model this project uses."""

    faces: np.ndarray  # (F, 3) int
    v_template: np.ndarray  # (778, 3)
    weights: np.ndarray  # (778, 16) skinning weights
    kintree_table: np.ndarray  # (2, 16)
    shapedirs: np.ndarray | None = None
    posedirs: np.ndarray | None = None
    J_regressor: np.ndarray | None = None
    #: Rest-pose joint locations, stored directly in the model file. Using
    #: these avoids needing J_regressor at all for the unshaped hand.
    J: np.ndarray | None = None
    hands_mean: np.ndarray | None = None
    hands_components: np.ndarray | None = None
    raw: dict[str, Any] | None = None

    @property
    def n_vertices(self) -> int:
        return len(self.v_template)

    @property
    def parents(self) -> np.ndarray:
        """Parent index per joint, with the root marked -1."""
        p = np.array(self.kintree_table[0], dtype=np.int64).copy()
        p[0] = -1
        return p

    def summary(self) -> str:
        rows = [
            f"vertices      {self.v_template.shape}",
            f"faces         {self.faces.shape}",
            f"skin weights  {self.weights.shape}",
            f"parents       {self.parents.tolist()}",
        ]
        for name in ("shapedirs", "posedirs", "J_regressor", "J", "hands_components"):
            arr = getattr(self, name)
            rows.append(f"{name:<13} {None if arr is None else arr.shape}")
        return "\n".join(rows)


def load_mano(path: str | Path) -> ManoModel:
    """Read a MANO ``.pkl`` into plain numpy arrays."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Register at https://mano.is.tue.mpg.de/ and "
            "extract MANO_RIGHT.pkl from mano_v1_2.zip."
        )
    with path.open("rb") as fh:
        raw = _Unpickler(io.BytesIO(fh.read()), encoding="latin1").load()

    data = {k: _unwrap(v) for k, v in raw.items()}

    def arr(key: str) -> np.ndarray | None:
        v = data.get(key)
        if v is None:
            return None
        if isinstance(v, tuple):  # sparse reconstruction tuple
            return None
        return np.asarray(v)

    faces = arr("f")
    v_template = arr("v_template")
    weights = arr("weights")
    if faces is None or v_template is None or weights is None:
        raise ValueError(
            f"{path} is missing expected keys; got {sorted(data)}. "
            "Is this really a MANO model file?"
        )

    return ManoModel(
        faces=np.asarray(faces, dtype=np.int64),
        v_template=np.asarray(v_template, dtype=np.float64),
        weights=np.asarray(weights, dtype=np.float64),
        kintree_table=np.asarray(arr("kintree_table"), dtype=np.int64),
        shapedirs=arr("shapedirs"),
        posedirs=arr("posedirs"),
        J_regressor=arr("J_regressor"),
        J=arr("J"),
        hands_mean=arr("hands_mean"),
        hands_components=arr("hands_components"),
        raw=data,
    )
