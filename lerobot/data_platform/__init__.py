"""Data Platform domains, with lazy compatibility for historical module imports."""

from ._compat import install_aliases, legacy_attribute

install_aliases()


def __getattr__(name):
    module = legacy_attribute(name)
    globals()[name] = module
    return module
