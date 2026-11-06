"""Compatibility import for the relocated data-platform precompute package."""

from lerobot.data_platform.operators import implementations as _implementation
from lerobot.data_platform.operators.implementations import *  # noqa: F403

__path__ = _implementation.__path__
__all__ = _implementation.__all__
