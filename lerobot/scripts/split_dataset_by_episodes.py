#!/usr/bin/env python3
"""Compatibility entry point for the data-platform dataset splitter."""

import sys

from lerobot.data_platform.operators.implementations.preprocess import split_cli as _implementation

if __name__ == "__main__":
    _implementation.main()
else:
    sys.modules[__name__] = _implementation
