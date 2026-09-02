#!/bin/sh
# Build the conda environment that runs dex-retargeting's DexPilot optimizer.
#
# Separate from the project's uv venv on purpose: dex-retargeting needs
# Pinocchio, which has no working pip wheel here, and the venv's torch is pinned
# to the cu128 index that CLAUDE.md says not to disturb.
#
# The one line that matters is the numpy force-reinstall. conda-forge's numpy on
# this machine cannot multiply matrices -- `np.eye(3).dot(np.eye(3))` crashes the
# interpreter silently, exit 127, no traceback. Every downstream failure traced
# back to it: Pinocchio's URDF build, pytransform3d's matrix_from_euler, and
# dex_retargeting's config build were all just that one dot product. Pinocchio
# itself is fine and loads the Shadow Hand URDF standalone (nq=24). Replacing
# numpy with the pip build fixes all of them.
#
# Recorded because two hours went into concluding, wrongly, that this needed
# WSL and that Pinocchio's Windows build was at fault.
set -e
export PATH="/c/Users/maurice/miniforge3:/c/Users/maurice/miniforge3/Scripts:/c/Users/maurice/miniforge3/condabin:$PATH"
ENV=dexret2
E="C:/Users/maurice/miniforge3/envs/$ENV"

conda create -y -q -n "$ENV" -c conda-forge \
    python=3.11 pinocchio pytorch-cpu trimesh lxml networkx scipy pyyaml anytree

# Order matters: numpy first, so nothing installed after it pulls the broken one
# back in.
"$E/python.exe" -m pip install --quiet --force-reinstall --no-deps numpy
"$E/python.exe" -m pip install --quiet --no-deps dex-retargeting pytransform3d anytree
"$E/python.exe" -m pip install --quiet nlopt

"$E/python.exe" -c "
import numpy as np
assert np.eye(3).dot(np.eye(3)).sum() == 3.0, 'BLAS still broken'
from dex_retargeting.retargeting_config import RetargetingConfig
import dex_retargeting, pathlib
RetargetingConfig.set_default_urdf_dir(r'D:\datasets\dex-urdf\robots\hands')
cfg = RetargetingConfig.load_from_file(
    pathlib.Path(dex_retargeting.__file__).parent / 'configs/teleop/shadow_hand_right_dexpilot.yml')
r = cfg.build()
print('ok:', type(r.optimizer).__name__, len(r.joint_names), 'joints')
"
