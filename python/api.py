"""Module-style Python API wrapper over the compiled `ag` bindings.

Provides a single `api` object that exposes a compact, discoverable Python-facing
interface for tests and examples. Backwards-compatible top-level helper
functions are kept but delegate to `api`.
"""
import numpy as np
import ag as _ag


class API:
    """High-level, Python-facing API over the compiled `ag` bindings.

    Usage:
      from python.api import api
      A = api.var_from_numpy(...)
      C = api.matmul(A, B)

    The object intentionally mirrors a small subset of NumPy-style helpers and
    re-exports some low-level bindings where useful.
    """

    def __init__(self):
        self.backend = _ag
        self.Variable = _ag.Variable

    # --- conversion helpers ---
    def to_numpy(self, var):
        vals = np.array(var.value(), dtype=np.float32)
        return vals.reshape(tuple(var.shape()))

    def grad_to_numpy(self, var):
        g = np.array(var.grad(), dtype=np.float32)
        return g.reshape(tuple(var.shape()))

    def var_from_numpy(self, arr, requires_grad=False):
        a = np.array(arr, dtype=np.float32, copy=False)
        return self.Variable.from_numpy(a, requires_grad)

    def zeros_like(self, var, requires_grad=False):
        s = tuple(var.shape())
        return self.var_from_numpy(np.zeros(s, dtype=np.float32), requires_grad)

    # --- ops forwarded to backend ---
    def matmul(self, A, B):
        return _ag.matmul(A, B)

    def transpose(self, A):
        return A.T

    def at(self, A, begin, end):
        return A.at(begin, end)

    def reshape(self, A, new_shape):
        return _ag.reshape(A, new_shape)

    def reduce_sum(self, x, axes=None, keepdims=False):
        return _ag.reduce_sum(x, axes if axes is not None else [], keepdims)

    def flatten(self, x, start_dim=1):
        return _ag.flatten(x, start_dim)

    def nograd(self):
        return _ag.nograd()

    # --- direct passthroughs for convenience ---
    def is_grad_enabled(self):
        return _ag.is_grad_enabled()

    def set_grad_enabled(self, enabled: bool):
        return _ag.set_grad_enabled(enabled)


# Single instance used by tests / examples
api = API()

# Back-compat: simple functions that delegate to `api` so older tests still work
def to_numpy(var):
    return api.to_numpy(var)


def grad_to_numpy(var):
    return api.grad_to_numpy(var)


def var_from_numpy(arr, requires_grad=False):
    return api.var_from_numpy(arr, requires_grad)


def zeros_like(var, requires_grad=False):
    return api.zeros_like(var, requires_grad)


def matmul(A, B):
    return api.matmul(A, B)


def transpose(A):
    return api.transpose(A)


def at(A, begin, end):
    return api.at(A, begin, end)


def reshape(A, new_shape):
    return api.reshape(A, new_shape)


def reduce_sum(x, axes=None, keepdims=False):
    return api.reduce_sum(x, axes, keepdims)


def flatten(x, start_dim=1):
    return api.flatten(x, start_dim)


def nograd():
    return api.nograd()


__all__ = [
    "api",
    "to_numpy",
    "grad_to_numpy",
    "var_from_numpy",
    "zeros_like",
    "matmul",
    "transpose",
    "at",
    "reshape",
    "reduce_sum",
    "flatten",
    "nograd",
]
