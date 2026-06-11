"""
``numpy.lib`` is mostly a space for implementing functions that don't
belong in core or in another NumPy submodule with a clear purpose
(e.g. ``random``, ``fft``, ``linalg``, ``ma``).

``numpy.lib``'s private submodules contain basic functions that are used by
other public modules and are useful to have in the main name-space.

"""

# PEP 810: deferred on Python 3.15+, ignored on older Python.
__lazy_modules__ = [
    "numpy.lib._arraypad_impl",
    "numpy.lib._arraysetops_impl",
    "numpy.lib._function_base_impl",
    "numpy.lib._histograms_impl",
    "numpy.lib._index_tricks_impl",
    "numpy.lib._nanfunctions_impl",
    "numpy.lib._npyio_impl",
    "numpy.lib._polynomial_impl",
    "numpy.lib._shape_base_impl",
    "numpy.lib._stride_tricks_impl",
    "numpy.lib._twodim_base_impl",
    "numpy.lib._type_check_impl",
    "numpy.lib._ufunclike_impl",
    "numpy.lib._utils_impl",
    "numpy.lib.scimath",
]

# Public submodules
# Note: recfunctions is public, but not imported
from numpy._core._multiarray_umath import add_docstring, tracemalloc_domain
from numpy._core.function_base import add_newdoc

# Private submodules.  Originally this block did
# ``from . import (_arraypad_impl, _arraysetops_impl, ...)`` to register the
# submodule names (see https://github.com/networkx/networkx/issues/5838).
# That form is *not* affected by PEP 810's ``__lazy_modules__``: ``from . import
# submodule`` always loads the submodule eagerly even when the name is listed
# in ``__lazy_modules__``.  We split it so the lazy modules can actually defer:
#   * the modules in ``__lazy_modules__`` above are accessed on demand via
#     Python's normal attribute lookup (and the explicit ``lazy from . import``
#     statements further down for Python 3.15+);
#   * the always-eager modules below stay eager.
from . import (
    _arrayterator_impl,
    _version,
    array_utils,
    format,
    introspect,
    mixins,
    npyio,
    stride_tricks,
)

# numpy.lib namespace members
from ._arrayterator_impl import Arrayterator
from ._version import NumpyVersion

__all__ = [
    "Arrayterator", "add_docstring", "add_newdoc", "array_utils",
    "format", "introspect", "mixins", "NumpyVersion", "npyio", "scimath",
    "stride_tricks", "tracemalloc_domain",
]

add_newdoc.__module__ = "numpy.lib"

from numpy._pytesttester import PytestTester

test = PytestTester(__name__)
del PytestTester

def __getattr__(attr):
    # Warn for deprecated/removed aliases
    import warnings

    if attr == "emath":
        raise AttributeError(
            "numpy.lib.emath was an alias for emath module that was removed "
            "in NumPy 2.0. Replace usages of numpy.lib.emath with "
            "numpy.emath.",
            name=None
        )
    elif attr in (
        "histograms", "type_check", "nanfunctions", "function_base",
        "arraypad", "arraysetops", "ufunclike", "utils", "twodim_base",
        "shape_base", "polynomial", "index_tricks",
    ):
        raise AttributeError(
            f"numpy.lib.{attr} is now private. If you are using a public "
            "function, it should be available in the main numpy namespace, "
            "otherwise check the NumPy 2.0 migration guide.",
            name=None
        )
    elif attr == "arrayterator":
        raise AttributeError(
            "numpy.lib.arrayterator submodule is now private. To access "
            "Arrayterator class use numpy.lib.Arrayterator.",
            name=None
        )
    else:
        raise AttributeError(f"module {__name__!r} has no attribute {attr!r}")
