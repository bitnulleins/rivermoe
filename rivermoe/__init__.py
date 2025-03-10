from rivermoe import classification, regression, utils

__all__ = ["classification", "regression", "utils"]

import sys
from importlib import metadata as importlib_metadata


def get_version() -> str:
    try:
        return importlib_metadata.version(__name__)
    except importlib_metadata.PackageNotFoundError:  # pragma: no cover
        return "unknown"


__version__: str = get_version()
