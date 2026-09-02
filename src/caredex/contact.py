"""Contact descriptors: what a hand pose *touches*, not where its joints are.

Why this module exists
----------------------
MotionBricks' actual contribution is not that its latent space is modular --
mixtures of experts are old. It is the **authoring interface**: a user supplies
sparse intent (proxy keyframes, a navigation goal) and the backbone fills in
approach, contact and follow-through. Composition is something you *specify*,
not something you hope falls out of a bottleneck.

Proxy keyframes do not transfer to hands. Hand manipulation is not a
where-does-the-end-effector-go problem; it is a which-surfaces-touch-with-what-
force problem. The hand analogue of MotionBricks' interface is therefore a
**contact specification**, and that is what this module computes.

The second-round prior-art search found exactly this cell open: contact-mode
planning (Chavan-Dafle, Rodriguez, Cheng) indexes primitives by *discrete*
contact topology (stick / slip / separate), APriCoT by contact-state
transitions, IMCopilot by task-level goals. A *rich continuous* contact
parameterisation -- which patches, how much area, what normal and tangential
direction -- used as the native control interface was not found.

What a descriptor contains
--------------------------
Given a posed hand mesh and a probe surface standing in for the thing being
touched, per hand region:

    contact fraction   share of that region's vertices within `eps` of the probe
    contact area       approximate touching area in m^2
    mean depth         how far inside the probe the region reaches
    normal alignment   how squarely the region's surface faces the probe

Plus whole-hand aggregates: total area, number of regions engaged, and the
opposition axis (the dominant direction between engaged regions), which is what
distinguishes a pinch from a palmar press at equal total area.

Honest limits
-------------
* **Geometry, not force.** OakInk and DexYCB annotate kinematics only, so there
  is no force supervision anywhere in this project. "Contact area" here is a
  geometric quantity; turning it into pressure needs the Phase 3 simulator, and
  any pressure number before then is invented.
* **The probe is an approximation** of whatever is really being touched. For a
  caregiving cloth task a local plane is a fair model of fabric and skin; for a
  grasped rigid object it is a bounding proxy and should be described as one.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from caredex.mesh_collision import DIGIT_NAMES, digit_of_vertices
from caredex.mano import ManoModel

#: Finer than digit level: the pads that actually do manipulation are the
#: distal segments, and lumping them with the proximal phalanges hides the
#: difference between a fingertip pinch and a whole-finger wrap.
REGION_NAMES: tuple[str, ...] = (
    "palm",
    "index_prox", "index_dist",
    "middle_prox", "middle_dist",
    "ring_prox", "ring_dist",
    "pinky_prox", "pinky_dist",
    "thumb_prox", "thumb_dist",
)


@dataclass
class ProbeSurface:
    """The thing being touched.

    ``kind`` is ``"plane"``, ``"sphere"`` or ``"cylinder"``. A plane is the
    right model for cloth and skin, which is the Phase 3 case; the other two
    approximate grasped objects.
    """

    kind: str = "plane"
    origin: np.ndarray = field(default_factory=lambda: np.zeros(3))
    #: Plane normal, or cylinder axis. Ignored for spheres.
    axis: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 1.0]))
    radius: float = 0.03

    def signed_distance(self, points: np.ndarray) -> np.ndarray:
        """Signed distance, negative inside the probe."""
        p = points - self.origin
        if self.kind == "plane":
            n = self.axis / max(np.linalg.norm(self.axis), 1e-9)
            return p @ n
        if self.kind == "sphere":
            return np.linalg.norm(p, axis=-1) - self.radius
        if self.kind == "cylinder":
            a = self.axis / max(np.linalg.norm(self.axis), 1e-9)
            along = p @ a
            radial = p - along[:, None] * a
            return np.linalg.norm(radial, axis=-1) - self.radius
        raise ValueError(f"unknown probe kind {self.kind!r}")

    def outward_normal(self, points: np.ndarray) -> np.ndarray:
        """Unit normal of the probe surface at the nearest point to each input."""
        p = points - self.origin
        if self.kind == "plane":
            n = self.axis / max(np.linalg.norm(self.axis), 1e-9)
            return np.broadcast_to(n, points.shape).copy()
        if self.kind == "sphere":
            d = np.linalg.norm(p, axis=-1, keepdims=True)
            return p / np.maximum(d, 1e-9)
        a = self.axis / max(np.linalg.norm(self.axis), 1e-9)
        radial = p - (p @ a)[:, None] * a
        d = np.linalg.norm(radial, axis=-1, keepdims=True)
        return radial / np.maximum(d, 1e-9)


@dataclass
class ContactDescriptor:
    """Per-region contact state plus whole-hand aggregates."""

    fraction: np.ndarray          # (11,) share of region vertices in contact
    area: np.ndarray              # (11,) m^2
    depth: np.ndarray             # (11,) metres, positive means pressing in
    alignment: np.ndarray         # (11,) cos angle between region and probe normals
    total_area: float
    n_regions: int
    opposition: np.ndarray        # (3,) unit axis between engaged regions, zeros if <2

    def to_vector(self) -> np.ndarray:
        """Flat ``(47,)`` encoding, the form the model consumes."""
        return np.concatenate(
            [self.fraction, self.area * 1e3, self.depth * 1e3, self.alignment,
             [self.total_area * 1e3], self.opposition]
        ).astype(np.float32)

    @staticmethod
    def vector_size() -> int:
        return 4 * len(REGION_NAMES) + 1 + 3

    def summary(self) -> str:
        rows = [f"total area {self.total_area * 1e4:.2f} cm^2 across {self.n_regions} regions"]
        for i, name in enumerate(REGION_NAMES):
            if self.fraction[i] > 0:
                rows.append(
                    f"  {name:<12} {self.fraction[i]:>5.1%}  "
                    f"{self.area[i] * 1e4:>6.2f} cm^2  depth {self.depth[i] * 1e3:>6.2f} mm  "
                    f"align {self.alignment[i]:>+5.2f}"
                )
        return "\n".join(rows)


class ContactExtractor:
    """Reusable extractor; precomputes region membership and vertex areas once."""

    def __init__(self, model: ManoModel, eps: float = 0.004) -> None:
        self.model = model
        #: Vertices within this distance of the probe count as touching. 4 mm is
        #: about the resolution the 778-vertex MANO mesh can resolve; smaller
        #: values make the descriptor mostly sampling noise.
        self.eps = eps
        self.region_of_vertex = _region_of_vertices(model)
        self.vertex_area = _vertex_areas(model)
        self.faces = model.faces

    def extract(self, vertices: np.ndarray, probe: ProbeSurface) -> ContactDescriptor:
        """Descriptor for one posed mesh, ``(778, 3)``."""
        sd = probe.signed_distance(vertices)
        touching = sd < self.eps

        normals = _vertex_normals(vertices, self.faces)
        probe_n = probe.outward_normal(vertices)
        align = np.sum(normals * probe_n, axis=-1)

        n_reg = len(REGION_NAMES)
        fraction = np.zeros(n_reg)
        area = np.zeros(n_reg)
        depth = np.zeros(n_reg)
        alignment = np.zeros(n_reg)

        for r in range(n_reg):
            mask = self.region_of_vertex == r
            if not mask.any():
                continue
            hit = mask & touching
            fraction[r] = hit.sum() / mask.sum()
            if hit.any():
                area[r] = self.vertex_area[hit].sum()
                depth[r] = float(np.maximum(-sd[hit], 0.0).mean())
                alignment[r] = float(align[hit].mean())

        engaged = np.flatnonzero(area > 0)
        opposition = np.zeros(3)
        if len(engaged) >= 2:
            # Axis between the two most-engaged regions' contact centroids. This
            # is what separates a two-pad pinch from a flat palmar press at the
            # same total area -- and for cloth work, a pinch and a press are
            # different primitives even when they touch equally much.
            order = engaged[np.argsort(-area[engaged])][:2]
            cents = []
            for r in order:
                hit = (self.region_of_vertex == r) & touching
                cents.append(vertices[hit].mean(axis=0))
            v = cents[0] - cents[1]
            n = np.linalg.norm(v)
            if n > 1e-9:
                opposition = v / n

        return ContactDescriptor(
            fraction=fraction,
            area=area,
            depth=depth,
            alignment=alignment,
            total_area=float(area.sum()),
            n_regions=int(len(engaged)),
            opposition=opposition,
        )


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------


def _region_of_vertices(model: ManoModel) -> np.ndarray:
    """Split each digit into proximal and distal halves by skinning weight.

    MANO joints 3, 6, 9, 12, 15 are the distal joints of index, middle, pinky,
    ring and thumb; a vertex is distal if its dominant weight sits there.
    """
    dominant = model.weights.argmax(axis=1)
    digit = digit_of_vertices(model)
    distal_joints = {3, 6, 9, 12, 15}

    out = np.zeros(len(dominant), dtype=np.int64)
    for i, (d, j) in enumerate(zip(digit, dominant)):
        name = DIGIT_NAMES[d]
        if name == "palm":
            out[i] = REGION_NAMES.index("palm")
        else:
            part = "dist" if int(j) in distal_joints else "prox"
            out[i] = REGION_NAMES.index(f"{name}_{part}")
    return out


def _vertex_areas(model: ManoModel) -> np.ndarray:
    """Barycentric vertex areas of the rest mesh, in m^2.

    Computed on the rest pose rather than per frame: skinning barely changes
    surface area, and recomputing it every frame would triple the cost of the
    descriptor for a sub-percent difference.
    """
    v = model.v_template
    f = model.faces
    tri = v[f]
    face_area = 0.5 * np.linalg.norm(
        np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=-1
    )
    out = np.zeros(len(v))
    np.add.at(out, f.ravel(), np.repeat(face_area / 3.0, 3))
    return out


def _vertex_normals(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    tri = vertices[faces]
    fn = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    out = np.zeros_like(vertices)
    np.add.at(out, faces.ravel(), np.repeat(fn, 3, axis=0))
    n = np.linalg.norm(out, axis=-1, keepdims=True)
    return out / np.maximum(n, 1e-9)


def fit_probe_plane(vertices: np.ndarray, region_mask: np.ndarray) -> ProbeSurface:
    """A plane fitted to the palmar side, standing in for a cloth or skin surface.

    Used to give existing kinematic-only datasets a probe at all: neither OakInk
    nor DexYCB records the fabric or skin a caregiving hand would touch, so the
    surface is inferred from the hand's own palmar geometry. This is a modelling
    choice, not a measurement, and any descriptor built on it inherits that.
    """
    pts = vertices[region_mask]
    centroid = pts.mean(axis=0)
    _, _, vt = np.linalg.svd(pts - centroid)
    normal = vt[2]
    # Orient the plane so the hand sits on its positive side.
    if (vertices - centroid) @ normal < 0:
        normal = -normal
    return ProbeSurface(kind="plane", origin=centroid, axis=normal)
