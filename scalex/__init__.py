# scalex/__init__.py

# flake8: noqa: F401, F403 (Ignore "unused import" and "import *" warnings)

# --- Explicitly import key components to be part of the public API ---
from .utils import ScaleXEnhancer

# Optionally, import main model and dataset classes if they should be top-level
from .models.scalex_model import ScaleXModel  # Adjust filename if needed
from .data.ffhq_degradation_dataset import FFHQDegradationDataset

# Import submodules for direct access if needed (e.g., scalex.archs)
# but also to ensure their __init__.py runs for registrations.
from . import archs
from . import data
from . import models

# from . import utils # If utils.py still exists and has public utilities

# --- Version Information ---
try:
    from .version import __version__, __gitsha__, version_info
except ImportError:
    __version__ = "0.0.0.unknown"
    __gitsha__ = "unknown"
    version_info = (
        0,
        0,
        0,
        "unknown",
        "unknown",
    )  # Add a placeholder for releaselevel if needed

# --- Public API Definition (`__all__`) ---
__all__ = [
    "ScaleXEnhancer",
    "ScaleXModel",
    "FFHQDegradationDataset",
    "archs",  # Makes scalex.archs accessible
    "data",  # Makes scalex.data accessible
    "models",  # Makes scalex.models accessible
    # 'utils', # If you have a utils module to export
    "__version__",
    "__gitsha__",
    "version_info",
]
