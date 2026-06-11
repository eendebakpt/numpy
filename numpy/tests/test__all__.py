
import collections
import importlib

import numpy as np


def test_no_duplicates_in_np__all__():
    # Regression test for gh-10198.
    dups = {k: v for k, v in collections.Counter(np.__all__).items() if v > 1}
    assert len(dups) == 0


def test_lazy_public_names_in_sync():
    """`numpy._lazy_public_names` must equal the union of `__all__` of every
    module in `numpy.__lazy_modules__` (plus the matrixlib aliases).

    The set is hardcoded in `numpy/__init__.py` to avoid forcing every lazy
    module to load just to read its `__all__`.  This test catches drift when
    a public name is added to or removed from an impl module.
    """
    # Modules whose `__all__` makes up `_lazy_public_names`.  `numpy.lib`
    # itself, `numpy.lib.scimath` and `numpy._array_api_info` are listed in
    # `__lazy_modules__` but do not contribute names (they are exposed as
    # `lib` / `emath` / `__array_namespace_info__` directly, not via
    # `_lazy_public_names`).
    contributors = [
        m for m in np.__lazy_modules__
        if m not in {"numpy.lib", "numpy.lib.scimath", "numpy._array_api_info"}
    ]
    expected = set()
    for modname in contributors:
        mod = importlib.import_module(modname)
        expected |= set(getattr(mod, "__all__", ()))

    actual = set(np._lazy_public_names)
    missing = expected - actual
    stale = actual - expected
    assert not missing and not stale, (
        "numpy._lazy_public_names has drifted from the union of __all__ "
        "of every numpy.__lazy_modules__ member.\n"
        f"  missing (add to _lazy_public_names): {sorted(missing)}\n"
        f"  stale   (remove from _lazy_public_names): {sorted(stale)}"
    )
