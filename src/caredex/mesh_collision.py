"""Mesh-level self-intersection for the MANO hand.

Replaces the capsule proxy in :mod:`caredex.kinematics` with actual geometry:
linear blend skinning to pose the mesh, then triangle-triangle intersection
between triangles belonging to different digits.

Why the proxy was not enough
----------------------------
A capsule model cannot see shallow interpenetration -- a fingertip pressing a
few millimetres into the side of an adjacent finger clears the capsule test
while being clearly wrong on the mesh. For a prior whose decoded poses feed an
RL policy that will be scored on skin contact, "the capsules did not overlap"
is a weaker statement than it sounds.

Both remain useful. The capsule test is ~100x cheaper and is the right choice
inside a training loop; this is the right choice for the validation report.

What is still approximate
-------------------------
* Triangles of the *same* digit are never tested against each other, so a
  finger folding into itself is invisible. Adjacent triangles always share
  edges, and separating genuine self-folding from shared-edge contact needs
  geodesic distance bookkeeping this does not do.
* The palm is one region, so palm-to-palm contact is not tested either.
* Contact and interpenetration are not distinguished: touching triangles count
  as intersecting. For a hand prior that is the conservative direction, but it
  means a firm grasp scores as an intersection.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from caredex.mano import ManoModel

#: MANO joint -> digit name. Joint 0 is the wrist, treated as palm.
_JOINT_DIGIT = {
    0: "palm",
    1: "index", 2: "index", 3: "index",
    4: "middle", 5: "middle", 6: "middle",
    7: "pinky", 8: "pinky", 9: "pinky",
    10: "ring", 11: "ring", 12: "ring",
    13: "thumb", 14: "thumb", 15: "thumb",
}


def digit_of_vertices(model: ManoModel) -> np.ndarray:
    """Assign each vertex to a digit, by its dominant skinning weight.

    Returns an integer array of shape ``(778,)`` indexing into
    :data:`DIGIT_NAMES`.
    """
    dominant = model.weights.argmax(axis=1)
    names = DIGIT_NAMES
    return np.array([names.index(_JOINT_DIGIT[int(j)]) for j in dominant], dtype=np.int64)


DIGIT_NAMES = ["palm", "index", "middle", "ring", "pinky", "thumb"]


def digit_of_faces(model: ManoModel) -> np.ndarray:
    """Digit index per triangle, by majority vote over its three vertices.

    Ties fall to the lowest index, which puts boundary triangles on the palm --
    the conservative choice, since palm triangles are excluded from far fewer
    pairs than finger triangles are.
    """
    vd = digit_of_vertices(model)
    tri = vd[model.faces]
    out = np.empty(len(tri), dtype=np.int64)
    for i, row in enumerate(tri):
        counts = np.bincount(row, minlength=len(DIGIT_NAMES))
        out[i] = int(counts.argmax())
    return out


# ---------------------------------------------------------------------------
# Linear blend skinning
# ---------------------------------------------------------------------------


def _rodrigues(axis_angle: np.ndarray) -> np.ndarray:
    """``(..., 3)`` axis-angle -> ``(..., 3, 3)`` rotation matrices."""
    theta = np.linalg.norm(axis_angle, axis=-1, keepdims=True)
    safe = np.where(theta < 1e-8, 1.0, theta)
    k = axis_angle / safe
    kx, ky, kz = k[..., 0], k[..., 1], k[..., 2]
    zero = np.zeros_like(kx)
    K = np.stack(
        [
            np.stack([zero, -kz, ky], -1),
            np.stack([kz, zero, -kx], -1),
            np.stack([-ky, kx, zero], -1),
        ],
        axis=-2,
    )
    eye = np.broadcast_to(np.eye(3), K.shape).copy()
    s = np.sin(theta)[..., None]
    c = np.cos(theta)[..., None]
    R = eye + s * K + (1 - c) * (K @ K)
    return np.where(theta[..., None] < 1e-8, eye, R)


def pose_mesh(
    model: ManoModel,
    joint_rotations: np.ndarray,
    betas: np.ndarray | None = None,
    apply_pose_blend: bool = True,
) -> np.ndarray:
    """Skin the MANO mesh. ``joint_rotations`` is ``(16, 3, 3)`` or ``(16, 3)``.

    Rotations are relative to each joint's parent, matching how OakInk and
    DexYCB store MANO pose. Returns ``(778, 3)`` posed vertices.
    """
    R = joint_rotations
    if R.shape[-1] == 3 and R.ndim == 2:  # axis-angle
        R = _rodrigues(R)
    if R.shape != (16, 3, 3):
        raise ValueError(f"expected (16, 3, 3) or (16, 3), got {joint_rotations.shape}")

    v = model.v_template.copy()
    if betas is not None and model.shapedirs is not None:
        v = v + model.shapedirs[..., : len(betas)] @ betas

    J = model.J_regressor @ v if model.J_regressor is not None else model.J
    if J is None:
        raise ValueError("model has neither J_regressor nor J")

    if apply_pose_blend and model.posedirs is not None:
        # MANO's pose blend shapes are driven by the non-root rotations with
        # the identity removed -- 15 joints x 9 entries = 135 features.
        feat = (R[1:] - np.eye(3)).reshape(-1)
        v = v + model.posedirs @ feat

    parents = model.parents
    T = np.zeros((16, 4, 4))
    T[:, 3, 3] = 1.0
    for j in range(16):
        local = np.eye(4)
        local[:3, :3] = R[j]
        local[:3, 3] = J[j] - (J[parents[j]] if parents[j] >= 0 else 0.0)
        T[j] = local if parents[j] < 0 else T[parents[j]] @ local

    # Remove the rest-pose offset so the rest pose maps to itself.
    for j in range(16):
        T[j][:3, 3] -= T[j][:3, :3] @ J[j]

    blended = np.einsum("vj,jab->vab", model.weights, T)
    homo = np.concatenate([v, np.ones((len(v), 1))], axis=1)
    return np.einsum("vab,vb->va", blended, homo)[:, :3]


# ---------------------------------------------------------------------------
# Triangle-triangle intersection
# ---------------------------------------------------------------------------


def _aabb(tris: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return tris.min(axis=1), tris.max(axis=1)


def _tri_tri_intersect(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Vectorised separating-axis test for triangle pairs.

    ``a`` and ``b`` are ``(N, 3, 3)``. Returns a boolean ``(N,)``. Uses the
    standard 11-axis SAT for triangles: the two face normals plus the nine
    edge-edge cross products. Coplanar pairs are reported as intersecting,
    which is the conservative direction.
    """
    n = len(a)
    if n == 0:
        return np.zeros(0, dtype=bool)

    ea = np.stack([a[:, 1] - a[:, 0], a[:, 2] - a[:, 1], a[:, 0] - a[:, 2]], axis=1)
    eb = np.stack([b[:, 1] - b[:, 0], b[:, 2] - b[:, 1], b[:, 0] - b[:, 2]], axis=1)
    na = np.cross(ea[:, 0], ea[:, 1])
    nb = np.cross(eb[:, 0], eb[:, 1])

    axes = [na[:, None, :], nb[:, None, :]]
    axes.append(np.cross(ea[:, :, None, :], eb[:, None, :, :]).reshape(n, 9, 3))
    axis = np.concatenate(axes, axis=1)  # (N, 11, 3)

    # Degenerate axes (parallel edges, zero-area triangles) cannot separate.
    norm = np.linalg.norm(axis, axis=-1)
    valid = norm > 1e-12

    pa = np.einsum("nkc,nvc->nkv", axis, a)
    pb = np.einsum("nkc,nvc->nkv", axis, b)
    separated = (pa.min(-1) > pb.max(-1) + 1e-12) | (pb.min(-1) > pa.max(-1) + 1e-12)
    return ~(separated & valid).any(axis=1)


@dataclass
class MeshCollisionReport:
    n_intersecting_pairs: int
    intersects: bool
    n_candidate_pairs: int
    digits_involved: list[tuple[str, str]]


class MeshSelfIntersection:
    """Reusable checker; precomputes the face-pair mask once per model."""

    def __init__(self, model: ManoModel, margin: float = 0.0, ring: int = 2) -> None:
        self.model = model
        self.margin = margin
        self.faces = model.faces
        self.face_digit = digit_of_faces(model)

        fd = self.face_digit
        i, j = np.triu_indices(len(fd), k=1)
        keep = fd[i] != fd[j]

        # Faces near each other on the surface always touch: triangles across a
        # digit boundary share edges, so testing them reports the mesh as
        # self-intersecting even in the open rest pose. Excluding topological
        # neighbours out to `ring` hops is what makes the test mean
        # "interpenetration" rather than "adjacency".
        neighbours = _face_neighbourhood(model.faces, ring)
        adjacent = np.array(
            [(a, b) in neighbours for a, b in zip(i[keep], j[keep])], dtype=bool
        )
        self.pair_i = i[keep][~adjacent]
        self.pair_j = j[keep][~adjacent]
        self.n_excluded_adjacent = int(adjacent.sum())

    def check(self, vertices: np.ndarray) -> MeshCollisionReport:
        """Test one posed mesh, ``(778, 3)``."""
        tris = vertices[self.faces]
        lo, hi = _aabb(tris)
        lo = lo - self.margin
        hi = hi + self.margin

        i, j = self.pair_i, self.pair_j
        # Broad phase: AABB overlap. This removes >99% of pairs, which is what
        # makes the exact test affordable at ~1.2M candidate pairs per frame.
        overlap = np.all((lo[i] <= hi[j]) & (lo[j] <= hi[i]), axis=1)
        ci, cj = i[overlap], j[overlap]

        hits = _tri_tri_intersect(tris[ci], tris[cj]) if len(ci) else np.zeros(0, bool)
        n_hits = int(hits.sum())

        pairs: list[tuple[str, str]] = []
        if n_hits:
            fd = self.face_digit
            seen = {
                tuple(sorted((DIGIT_NAMES[fd[a]], DIGIT_NAMES[fd[b]])))
                for a, b in zip(ci[hits], cj[hits])
            }
            pairs = sorted(seen)

        return MeshCollisionReport(
            n_intersecting_pairs=n_hits,
            intersects=n_hits > 0,
            n_candidate_pairs=int(overlap.sum()),
            digits_involved=pairs,
        )

    def rate(self, vertex_batch: np.ndarray) -> tuple[float, list[MeshCollisionReport]]:
        """Fraction of frames with any cross-digit intersection."""
        reports = [self.check(v) for v in vertex_batch]
        return float(np.mean([r.intersects for r in reports])), reports


class FastMeshProximity:
    """Cheap self-intersection screen sharing the mesh checker's geometry.

    Replaces the capsule proxy in :mod:`caredex.kinematics`, which was not
    merely miscalibrated but built on an FK convention that does not match
    MANO's joint frames: even after recalibrating from MANO's rest joints and
    shrinking radii to 2.5 mm it reported 26% interpenetration where the exact
    triangle test reported 5.5%. Two checks disagreeing by 5x is worse than one
    check, because the cheap one silently overrides judgement inside a training
    loop.

    This screen uses the same LBS-posed vertices as the exact test, decimated:
    for every pair of digits, the minimum vertex-to-vertex distance. Contact is
    declared below ``threshold``, which :meth:`calibrate` fits so that the
    screen's rate matches the exact checker's on real data.

    It is a screen, not a test: vertex sampling can step over a shallow
    intersection. Use it in training loops; use :class:`MeshSelfIntersection`
    for anything reported.
    """

    def __init__(self, model: ManoModel, n_per_digit: int = 96, threshold: float = 0.004) -> None:
        self.model = model
        self.threshold = threshold
        vd = digit_of_vertices(model)
        self.digits: dict[str, np.ndarray] = {}
        rng = np.random.default_rng(0)
        for i, name in enumerate(DIGIT_NAMES):
            idx = np.flatnonzero(vd == i)
            if len(idx) > n_per_digit:
                idx = rng.choice(idx, n_per_digit, replace=False)
            self.digits[name] = np.sort(idx)
        self.pairs = [
            (a, b)
            for i, a in enumerate(DIGIT_NAMES)
            for b in DIGIT_NAMES[i + 1 :]
            if len(self.digits[a]) and len(self.digits[b])
        ]

    def min_clearance(self, vertices: np.ndarray) -> float:
        """Smallest inter-digit vertex distance, in metres."""
        best = np.inf
        for a, b in self.pairs:
            va, vb = vertices[self.digits[a]], vertices[self.digits[b]]
            d = np.linalg.norm(va[:, None, :] - vb[None, :, :], axis=-1).min()
            best = min(best, float(d))
        return best

    def intersects(self, vertices: np.ndarray) -> bool:
        return self.min_clearance(vertices) < self.threshold

    def calibrate(self, vertex_batch: np.ndarray, target_rate: float) -> float:
        """Pick the threshold reproducing ``target_rate`` on these frames."""
        clearances = np.array([self.min_clearance(v) for v in vertex_batch])
        if not len(clearances):
            return self.threshold
        # The threshold is the target_rate-th quantile of clearance: below it,
        # exactly that fraction of frames is flagged.
        self.threshold = float(np.quantile(clearances, np.clip(target_rate, 0.0, 1.0)))
        return self.threshold


def dof_to_mano_rotations(
    q: np.ndarray, flexion_axis: int = 2, flexion_sign: float = 1.0
) -> np.ndarray:
    """Bridge the 27-DOF anatomical vector back to ``(16, 3, 3)`` MANO rotations.

    The inverse of the projection in :mod:`caredex.data.oakink`, and lossy in
    the same place: the anatomical model has no third rotational axis per
    joint, so that component is reconstructed as zero. Poses round-tripped
    through here are therefore a *subset* of what MANO can express -- which is
    the correct behaviour for validating a prior that only ever outputs
    anatomical DOF, but means this must not be used to re-derive ground truth.

    Defaults match what :func:`caredex.data.oakink.infer_flexion_axis` and
    :func:`~caredex.data.oakink.infer_flexion_sign` inferred from OakInk
    (z axis, positive flexion). Pass the values from ``bundle.meta`` when
    validating against a differently-converted dataset.
    """
    from caredex.data.oakink import MANO_JOINT_MAP
    from caredex.hand_model import DOF_INDEX, N_DOF

    q = np.asarray(q, dtype=np.float64)
    if q.shape[-1] != N_DOF:
        raise ValueError(f"expected trailing dim {N_DOF}, got {q.shape}")
    single = q.ndim == 1
    q = q.reshape(-1, N_DOF)

    abd_axis = (flexion_axis + 1) % 3
    euler = np.zeros((len(q), 16, 3))

    for joint, (digit, level) in MANO_JOINT_MAP.items():
        if digit == "thumb":
            names = {
                "cmc": ("thumb_cmc_flex", "thumb_cmc_abd"),
                "mcp": ("thumb_mcp_flex", "thumb_mcp_abd"),
                "ip": ("thumb_ip_flex", None),
            }[level]
        elif level == "mcp":
            names = (f"{digit}_mcp_flex", f"{digit}_mcp_abd")
        else:
            names = (f"{digit}_{level}_flex", None)

        euler[:, joint, flexion_axis] = q[:, DOF_INDEX[names[0]]] / flexion_sign
        if names[1] is not None:
            euler[:, joint, abd_axis] = q[:, DOF_INDEX[names[1]]]

    R = _euler_xyz_to_matrix(np.radians(euler))
    return R[0] if single else R


def _euler_xyz_to_matrix(e: np.ndarray) -> np.ndarray:
    """Intrinsic XYZ Euler angles (radians) -> rotation matrices.

    Inverse of :func:`caredex.data.oakink.matrix_to_euler_xyz`.
    """
    cx, cy, cz = np.cos(e[..., 0]), np.cos(e[..., 1]), np.cos(e[..., 2])
    sx, sy, sz = np.sin(e[..., 0]), np.sin(e[..., 1]), np.sin(e[..., 2])
    return np.stack(
        [
            np.stack([cy * cz, -cy * sz, sy], -1),
            np.stack([sx * sy * cz + cx * sz, -sx * sy * sz + cx * cz, -sx * cy], -1),
            np.stack([-cx * sy * cz + sx * sz, cx * sy * sz + sx * cz, cx * cy], -1),
        ],
        axis=-2,
    )


def mesh_interpenetration_rate(
    q: np.ndarray,
    model: ManoModel,
    checker: "MeshSelfIntersection | None" = None,
    flexion_axis: int = 2,
    flexion_sign: float = 1.0,
    max_frames: int | None = 256,
) -> dict:
    """Mesh-level self-intersection rate for a batch of 27-DOF poses.

    Roughly 70 ms per frame, so ``max_frames`` subsamples by default. Use the
    capsule proxy in :mod:`caredex.kinematics` when a cheap check is needed.
    """
    from caredex.hand_model import N_DOF

    q = np.asarray(q, dtype=np.float64).reshape(-1, N_DOF)
    if max_frames is not None and len(q) > max_frames:
        step = len(q) // max_frames
        q = q[::step][:max_frames]

    checker = checker or MeshSelfIntersection(model)
    rotations = dof_to_mano_rotations(q, flexion_axis, flexion_sign)

    hits, pair_counts = 0, []
    involved: set[tuple[str, str]] = set()
    for R in rotations:
        rep = checker.check(pose_mesh(model, R))
        hits += rep.intersects
        pair_counts.append(rep.n_intersecting_pairs)
        involved |= set(rep.digits_involved)

    return {
        "frames_checked": len(q),
        "mesh_interpenetration_rate": hits / max(len(q), 1),
        "mean_intersecting_pairs": float(np.mean(pair_counts)) if pair_counts else 0.0,
        "max_intersecting_pairs": int(np.max(pair_counts)) if pair_counts else 0,
        "digit_pairs_involved": sorted(involved),
    }


def _face_neighbourhood(faces: np.ndarray, ring: int) -> set[tuple[int, int]]:
    """Ordered face-index pairs within ``ring`` hops of vertex adjacency."""
    n_faces = len(faces)
    faces_of_vertex: dict[int, list[int]] = {}
    for f, tri in enumerate(faces):
        for v in tri:
            faces_of_vertex.setdefault(int(v), []).append(f)

    # One hop: faces sharing at least one vertex.
    adj: list[set[int]] = [set() for _ in range(n_faces)]
    for shared in faces_of_vertex.values():
        for a in shared:
            adj[a].update(shared)
    for f in range(n_faces):
        adj[f].discard(f)

    reach = [set(s) for s in adj]
    for _ in range(ring - 1):
        grown = []
        for f in range(n_faces):
            s = set(reach[f])
            for g in reach[f]:
                s |= adj[g]
            s.discard(f)
            grown.append(s)
        reach = grown

    pairs: set[tuple[int, int]] = set()
    for f, s in enumerate(reach):
        for g in s:
            pairs.add((f, g) if f < g else (g, f))
    return pairs
