"""From a :class:`TrajectoryBundle` to batched, normalised training windows."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from caredex.data.base import TrajectoryBundle
from caredex.hand_model import N_DOF, denormalize, normalize


@dataclass(frozen=True)
class SplitSizes:
    train: int
    val: int
    test: int


class WindowedTrajectoryDataset(Dataset):
    """Fixed-length windows cut from variable-length trajectories.

    Windows are indexed lazily over a precomputed ``(traj_idx, start)`` table,
    so no data is duplicated in memory. Poses are normalised into [-1, 1] by
    the joint limit box (:func:`caredex.hand_model.normalize`) rather than by
    dataset statistics -- the mapping must stay identical when the data source
    is swapped from synthetic to DexYCB, or a prior trained on one becomes
    meaningless on the other.
    """

    def __init__(
        self,
        trajectories: list[np.ndarray],
        window: int,
        stride: int = 1,
        labels: list[str] | None = None,
        aux: list[np.ndarray] | None = None,
    ) -> None:
        if window < 2:
            raise ValueError("window must be at least 2 frames")
        self.window = window
        self.stride = stride
        self.labels = labels or ["unlabelled"] * len(trajectories)

        # Auxiliary channels (GRAB's per-region contact) are appended after the
        # 27 pose dimensions, mapped from [0, 1] onto the same [-1, 1] range the
        # pose occupies. Without that rescaling a contact channel would carry
        # half the gradient weight of a joint angle purely because of its units,
        # and the whole point of the contact experiment is that the two are
        # compared on equal terms.
        self.n_pose = trajectories[0].shape[1]
        self.n_aux = 0 if aux is None else aux[0].shape[1]
        if aux is None:
            self.trajectories = [normalize(t) for t in trajectories]
        else:
            if len(aux) != len(trajectories):
                raise ValueError(f"{len(aux)} aux arrays for {len(trajectories)} trajectories")
            self.trajectories = [
                np.concatenate([normalize(t), 2.0 * a.astype(np.float32) - 1.0], axis=1)
                for t, a in zip(trajectories, aux)
            ]

        index: list[tuple[int, int]] = []
        for i, tr in enumerate(self.trajectories):
            if len(tr) < window:
                continue
            for s in range(0, len(tr) - window + 1, stride):
                index.append((i, s))
        if not index:
            raise ValueError(
                f"no trajectory is at least {window} frames long; "
                f"longest is {max((len(t) for t in trajectories), default=0)}"
            )
        self.index = np.array(index, dtype=np.int64)

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, i: int) -> torch.Tensor:
        traj_idx, start = self.index[i]
        w = self.trajectories[traj_idx][start : start + self.window]
        return torch.from_numpy(np.ascontiguousarray(w))

    def label_of(self, i: int) -> str:
        return self.labels[int(self.index[i][0])]


def split_bundle(
    bundle: TrajectoryBundle,
    val_frac: float = 0.1,
    test_frac: float = 0.1,
    seed: int = 0,
) -> tuple[TrajectoryBundle, TrajectoryBundle, TrajectoryBundle]:
    """Split by *trajectory*, never by frame.

    Splitting by frame would put windows from the same trajectory on both sides
    and inflate validation numbers -- consecutive frames of a 30 fps hand
    trajectory are nearly identical.
    """
    n = len(bundle.trajectories)
    if n < 3:
        raise ValueError(f"need at least 3 trajectories to split, got {n}")
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)

    n_test = max(1, int(round(test_frac * n)))
    n_val = max(1, int(round(val_frac * n)))
    if n_val + n_test >= n:
        raise ValueError(f"val+test fractions consume all {n} trajectories")

    parts = {
        "test": perm[:n_test],
        "val": perm[n_test : n_test + n_val],
        "train": perm[n_test + n_val :],
    }
    out = []
    for name in ("train", "val", "test"):
        idx = parts[name]
        out.append(
            TrajectoryBundle(
                trajectories=[bundle.trajectories[i] for i in idx],
                fps=bundle.fps,
                labels=[bundle.labels[i] for i in idx],
                meta={**bundle.meta, "split": name, "split_seed": seed},
            )
        )
    return out[0], out[1], out[2]


def make_dataloaders(
    bundle: TrajectoryBundle,
    window: int = 32,
    batch_size: int = 256,
    stride: int = 1,
    val_stride: int | None = None,
    val_frac: float = 0.1,
    test_frac: float = 0.1,
    seed: int = 0,
    num_workers: int = 0,
    device: torch.device | None = None,
) -> tuple[DataLoader, DataLoader, DataLoader, SplitSizes]:
    """Build train/val/test loaders with a trajectory-level split."""
    train_b, val_b, test_b = split_bundle(bundle, val_frac, test_frac, seed)
    vs = val_stride if val_stride is not None else max(1, window // 4)

    train_ds = WindowedTrajectoryDataset(train_b.trajectories, window, stride, train_b.labels)
    val_ds = WindowedTrajectoryDataset(val_b.trajectories, window, vs, val_b.labels)
    test_ds = WindowedTrajectoryDataset(test_b.trajectories, window, vs, test_b.labels)

    pin = device is not None and device.type == "cuda"
    common = dict(num_workers=num_workers, pin_memory=pin, drop_last=False)
    persist = dict(persistent_workers=num_workers > 0)

    return (
        DataLoader(train_ds, batch_size=batch_size, shuffle=True, **common, **persist),
        DataLoader(val_ds, batch_size=batch_size, shuffle=False, **common, **persist),
        DataLoader(test_ds, batch_size=batch_size, shuffle=False, **common, **persist),
        SplitSizes(len(train_ds), len(val_ds), len(test_ds)),
    )


def to_native(x: torch.Tensor) -> np.ndarray:
    """Normalised tensor -> native-unit numpy array (degrees / metres)."""
    arr = x.detach().cpu().numpy()
    if arr.shape[-1] != N_DOF:
        raise ValueError(f"expected trailing dim {N_DOF}, got {arr.shape}")
    return denormalize(arr)
