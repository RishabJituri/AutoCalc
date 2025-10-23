#!/usr/bin/env python3
"""Run pytests for this repository with correct PYTHONPATH so the compiled
`ag` extension (in the top-level `python/` directory) is importable.

Usage: python pytests/main.py [pytest-args...]
If no args are provided the tests in the `pytests` folder are executed.
"""
import sys
import os

# Ensure the repo root (containing ag/ package shim) is on sys.path before the
# compiled extension build dir so Python prefers the shim package. Also add the
# build/python directory afterward so ag._backend can be imported as a submodule
# when the extension is built into build/python/ag/_backend.*.so
HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(HERE)
AG_BUILD_DIR = os.path.join(REPO_ROOT, "python")
# Insert repo root first so local `ag/` package is found
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
# Then ensure compiled build dir is present so ag._backend can be imported
if AG_BUILD_DIR not in sys.path:
    sys.path.insert(1, AG_BUILD_DIR)


def main(argv=None):
    argv = argv if argv is not None else sys.argv[1:]
    try:
        import pytest
    except Exception as e:
        print("ERROR: pytest is not installed. Install it with: pip install pytest", file=sys.stderr)
        return 2

    # Default to running tests in the pytests folder if no args given
    if not argv:
        argv = ["-q", os.path.join(REPO_ROOT, "pytests")]

    # Run pytest with the provided args
    return pytest.main(argv)

if __name__ == '__main__':
    rc = main()
    sys.exit(rc)
