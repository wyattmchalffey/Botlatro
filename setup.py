"""Build configuration for Cython extensions.

The project's main packaging metadata lives in `pyproject.toml`; this
file exists only to register the Cython extension module(s). When
Cython isn't installed, the extension build is skipped silently so the
Python-only path still works.

Build command (from the project root, with the dev environment active):
    python -m pip install -e .

The native module is `balatro_ai.rules.hand_evaluator_native`. Its
source is `src/balatro_ai/rules/hand_evaluator_native.pyx`. The
extension is loaded with a try/except in `hand_evaluator.py` so the
project still runs in pure-Python mode if the compile fails.
"""

from __future__ import annotations

from setuptools import setup

try:
    from Cython.Build import cythonize
    _ext_modules = cythonize(
        ["src/balatro_ai/rules/hand_evaluator_native.pyx"],
        compiler_directives={"language_level": "3", "boundscheck": False, "wraparound": False},
    )
except ImportError:
    # Cython not installed → skip the native extension; pure-Python path
    # still works. The CI/dev environment installs Cython explicitly.
    _ext_modules = []


setup(ext_modules=_ext_modules)
