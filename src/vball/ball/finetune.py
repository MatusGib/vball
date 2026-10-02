"""Fine-tune the vendored TrackNetV3 on pseudo-labelled windows."""

import importlib
import sys
from types import ModuleType

from vball.config import TRACKNET_DIR


def tracknet_modules() -> tuple[ModuleType, ModuleType]:
    """Import the vendored TrackNetV3 `utils.general` and `dataset` modules."""
    path = str(TRACKNET_DIR)
    if path not in sys.path:
        sys.path.insert(0, path)
    return importlib.import_module("utils.general"), importlib.import_module("dataset")
