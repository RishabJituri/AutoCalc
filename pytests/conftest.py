import os

def pytest_ignore_collect(path, config):
    # Ignore top-level duplicate test files in pytests/ (we keep tests in pytests/test/)
    try:
        name = os.path.basename(str(path))
        parent = os.path.dirname(str(path))
        this_dir = os.path.dirname(__file__)
    except Exception:
        return False
    # Only ignore files that are directly in the pytests/ directory (not subfolders)
    if parent == this_dir and name in ("test_slice.py", "test_transpose.py"):
        return True
    return False
