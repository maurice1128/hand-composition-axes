from caredex.data.base import TrajectorySource, TrajectoryBundle, register_source, get_source
from caredex.data.synthetic import SyntheticSource

# Adapters for the real datasets are registered on import; they raise a helpful
# error at load() time if the data has not been downloaded yet.
from caredex.data import dexycb as _dexycb  # noqa: F401
from caredex.data import oakink as _oakink  # noqa: F401
from caredex.data import oakink2 as _oakink2  # noqa: F401
from caredex.data import grab as _grab  # noqa: F401
from caredex.data import taco as _taco  # noqa: F401

__all__ = [
    "TrajectorySource",
    "TrajectoryBundle",
    "register_source",
    "get_source",
    "SyntheticSource",
]
