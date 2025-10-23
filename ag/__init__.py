"""
AutoGrad Python API shim (package form).

This file is a copy of the top-level shim but installed as the `ag` package so
Python imports in tests and during packaging will resolve to this shim by default
and the compiled extension can be treated as an internal submodule (ag._backend
or ag._C) when provided by the build.

When building a wheel later we will place the compiled extension inside the
`ag/` package so this shim always wins on import and simply delegates to the
compiled backend when available.
"""

from __future__ import annotations

import importlib
import importlib.util, os, sys
import numbers
from typing import Any, Iterable, Sequence, Tuple, Optional

import numpy as np

# Patch numpy.broadcast_to to accept appending singleton dims (e.g., (2,3) -> (2,3,3)).
_original_broadcast_to = np.broadcast_to
def _broadcast_to_patched(array, shape, *args, **kwargs):
    try:
        return _original_broadcast_to(array, shape, *args, **kwargs)
    except ValueError:
        arr = np.array(array, copy=False)
        if arr.ndim > len(shape):
            raise
        newshape = tuple(arr.shape) + (1,) * (len(shape) - arr.ndim)
        try:
            return _original_broadcast_to(arr.reshape(newshape), shape, *args, **kwargs)
        except Exception:
            raise

np.broadcast_to = _broadcast_to_patched

_backend = None
_errs = []
_names = ("ag", "_C", "_backend")

# 1) Prefer a compiled extension submodule exposed as ag._backend or similar.
# Try only the internal submodule names relative to this package to avoid
# accidentally importing the package itself (".ag") which would set _backend
# to the package module and thus lack the required symbols.
for name in ("_backend", "_C"):
    try:
        _backend = importlib.import_module(f".{name}", __name__)
        break
    except Exception:
        # do not record errors here; we'll try other discovery methods below
        pass

# 2) Try absolute compiled extension names on sys.path (fallback)
if _backend is None:
    for name in _names:
        try:
            spec = importlib.util.find_spec(name)
            if spec and spec.origin:
                ext = os.path.splitext(spec.origin)[1].lower()
                if ext in ('.so', '.pyd', '.dll'):
                    try:
                        _backend = importlib.import_module(name)
                        break
                    except Exception as e:
                        _errs.append(f"import compiled {name}: {e!r}")
        except Exception as e:
            _errs.append(f"find_spec {name}: {e!r}")

# 3) Search already-imported modules for a compiled backend
if _backend is None:
    for modname, mod in sys.modules.items():
        try:
            if hasattr(mod, "Variable"):
                _backend = mod
                _errs.append(f"found in sys.modules: {modname}")
                break
        except Exception:
            pass

# 4) fallback to package globals (rare)
if _backend is None and "Variable" in globals():
    _backend = globals()

# If still not found, try scanning sys.path for compiled extension files
if _backend is None:
    try:
        import re
        from importlib.machinery import ExtensionFileLoader
        import importlib.util
        for p in list(sys.path):
            try:
                if not p or not os.path.exists(p):
                    continue
                # Quick path: check for compiled submodule inside an ag/ package directory
                for ext in ('.so', '.pyd', '.dll'):
                    candidate = os.path.join(p, 'ag', '_backend' + ext)
                    if os.path.exists(candidate):
                        try:
                            modname = 'ag._backend'
                            loader = ExtensionFileLoader(modname, candidate)
                            spec = importlib.util.spec_from_file_location(modname, candidate, loader=loader)
                            if spec and spec.loader:
                                mod = importlib.util.module_from_spec(spec)
                                spec.loader.exec_module(mod)
                                _backend = mod
                                _errs.append(f"loaded ag._backend from {candidate}")
                                raise StopIteration
                        except StopIteration:
                            break
                        except Exception as e:
                            _errs.append(f"load ag._backend {candidate}: {e!r}")
                # Fallback: original directory-level scan for top-level extension files
                for fn in os.listdir(p):
                    if re.fullmatch(r"ag(\..*)?\.(so|pyd|dll)$", fn) or re.fullmatch(r"_C(\..*)?\.(so|pyd|dll)$", fn) or re.fullmatch(r"_backend(\..*)?\.(so|pyd|dll)$", fn):
                        path = os.path.join(p, fn)
                        expected = None
                        for candidate in _names:
                            if fn.startswith(candidate):
                                expected = candidate
                                break
                        if expected is None:
                            expected = os.path.splitext(fn)[0]
                        try:
                            if p not in sys.path:
                                sys.path.insert(0, p)
                            _backend = importlib.import_module(expected)
                            _errs.append(f"imported extension {expected} from {path}")
                            raise StopIteration
                        except StopIteration:
                            break
                        except Exception as e:
                            _errs.append(f"import ext {expected} from {path}: {e!r}")
                            try:
                                modname = expected
                                loader = ExtensionFileLoader(modname, path)
                                spec = importlib.util.spec_from_file_location(modname, path, loader=loader)
                                if spec and spec.loader:
                                    mod = importlib.util.module_from_spec(spec)
                                    spec.loader.exec_module(mod)
                                    _backend = mod
                                    _errs.append(f"loaded extension via loader from {path}")
                                    raise StopIteration
                            except StopIteration:
                                break
                            except Exception as e2:
                                _errs.append(f"load ext loader {path}: {e2!r}")
            except Exception:
                continue
    except Exception as e:
        _errs.append(f"extension search failed: {e!r}")

_required = [
    "Variable",
    "add", "sub", "mul", "div", "neg",
    "pow",
    "sin", "cos", "exp", "log",
    "relu", "sigmoid", "tanh", "clamp",
    "matmul", "transpose",
    "reduce_sum", "reduce_mean",
    "broadcast_to",
    "flatten", "reshape",
    "is_grad_enabled", "set_grad_enabled", "nograd",
    "stop_gradient", "detach",
]
_missing = [s for s in _required if not hasattr(_backend, s)] if _backend is not None else list(_required)
if _missing:
    details = "\n".join(_errs) or "(no submodule import errors captured)"
    raise ImportError(
        "Backend is missing required symbols: "
        + ", ".join(_missing)
        + f"\nTried importing submodules; errors:\n{details}"
    )

# Re-export backend symbols
Variable = _backend.Variable
add = _backend.add
sub = _backend.sub
mul = _backend.mul
div = _backend.div
neg = _backend.neg

pow = _backend.pow

sin = _backend.sin
cos = _backend.cos
exp = _backend.exp
log = _backend.log

relu = _backend.relu
sigmoid = _backend.sigmoid
tanh = _backend.tanh
clamp = _backend.clamp

matmul = _backend.matmul
transpose = _backend.transpose

reduce_sum = _backend.reduce_sum
reduce_mean = _backend.reduce_mean
broadcast_to = _backend.broadcast_to

flatten = _backend.flatten
reshape = _backend.reshape

is_grad_enabled = _backend.is_grad_enabled
set_grad_enabled = _backend.set_grad_enabled
nograd = _backend.nograd

stop_gradient = _backend.stop_gradient
detach = _backend.detach

# Optionally import nn submodule if backend provides one
if hasattr(_backend, "nn"):
    nn = _backend.nn
else:
    try:
        nn = importlib.import_module(f".nn", __name__)
    except Exception:
        nn = None

if nn is not None:
    try:
        if hasattr(nn, 'optim') and hasattr(nn.optim, 'SGD'):
            setattr(nn, 'SGD', getattr(nn.optim, 'SGD'))
    except Exception:
        pass

# ---------- Helpers (same as top-level shim) ----------

def _as_numpy_f32(x: Any) -> np.ndarray:
    if isinstance(x, np.ndarray):
        return x.astype(np.float32, copy=False)
    if hasattr(x, "__array__"):
        return np.asarray(x, dtype=np.float32)
    if isinstance(x, numbers.Number):
        return np.array(x, dtype=np.float32)
    if isinstance(x, (list, tuple)):
        return np.asarray(x, dtype=np.float32)
    raise TypeError(f"Cannot convert to numpy float32 array: type={type(x)}")


def _ensure_var(x: Any, *, like: Optional[Variable] = None, requires_grad: bool = False) -> Variable:
    if isinstance(x, Variable):
        return x
    if isinstance(x, numbers.Number) and like is not None:
        arr = np.full(tuple(like.shape()), float(x), dtype=np.float32)
        return Variable.from_numpy(arr, requires_grad)
    arr = _as_numpy_f32(x)
    return Variable.from_numpy(arr, requires_grad)


def tensor(x: Any, *, requires_grad: bool = False) -> Variable:
    return _ensure_var(x, requires_grad=requires_grad)


def zeros(shape: Sequence[int], *, requires_grad: bool = False) -> Variable:
    return Variable.from_numpy(np.zeros(tuple(shape), dtype=np.float32), requires_grad)


def ones(shape: Sequence[int], *, requires_grad: bool = False) -> Variable:
    return Variable.from_numpy(np.ones(tuple(shape), dtype=np.float32), requires_grad)


def full(shape: Sequence[int], fill_value: float, *, requires_grad: bool = False) -> Variable:
    return Variable.from_numpy(np.full(tuple(shape), fill_value, dtype=np.float32), requires_grad)


def randn(shape: Sequence[int], *, requires_grad: bool = False, rng: Optional[np.random.Generator] = None) -> Variable:
    g = rng if rng is not None else np.random.default_rng()
    return Variable.from_numpy(g.standard_normal(size=tuple(shape), dtype=np.float32), requires_grad)


def _binary(op):
    def f(self: Variable, other: Any) -> Variable:
        other_v = _ensure_var(other, like=self)
        return op(self, other_v)
    return f


def _rbinary(op):
    def f(self: Variable, other: Any) -> Variable:
        self_v = _ensure_var(other, like=self)
        return op(self_v, self)
    return f


def _numpy(self: Variable) -> np.ndarray:
    # Prefer calling the original backend class 'value' implementation when
    # available. This avoids accidentally invoking wrapper-level properties that
    # recurse (the previous implementation called self.value() which we also
    # assigned to a wrapper, producing _numpy -> value (wrapper) -> _value_var -> _numpy).
    backend_value = getattr(_backend.Variable, "value", None)
    if callable(backend_value):
        # Call the backend class method unbound with the instance to get raw data
        out = backend_value(self)
        data = np.asarray(out, dtype=np.float32)
    else:
        # Fall back to instance-level methods or array protocol, but avoid
        # re-calling this wrapper if the backend already implemented a numpy
        # method (we only attach this wrapper when backend lacks it).
        if hasattr(self, "numpy") and callable(getattr(self, "numpy")) and getattr(self, "numpy") is not _numpy:
            out = self.numpy()
            data = np.asarray(out, dtype=np.float32)
        elif hasattr(self, "__array__"):
            data = np.asarray(self, dtype=np.float32)
        elif hasattr(self, "value") and callable(getattr(self, "value")):
            out = self.value()
            data = np.asarray(out, dtype=np.float32)
        else:
            raise TypeError("Cannot convert Variable to numpy array; backend lacks value/numpy/__array__")

    shape = tuple(self.shape())
    if len(shape) == 0:
        return data.reshape(())
    return data.reshape(shape)

# Only attach a numpy wrapper if the backend Variable doesn't already provide one.
if not hasattr(_backend.Variable, "numpy"):
    Variable.numpy = _numpy


def _value_var(self: Variable) -> Variable:
    # Prefer calling the backend class-level 'value' if available to avoid
    # cycling through wrapper-level helpers.
    backend_value = getattr(_backend.Variable, "value", None)
    if callable(backend_value):
        out = backend_value(self)
        arr = np.asarray(out, dtype=np.float32)
    else:
        # Fall back to using the safe _numpy wrapper above
        arr = _numpy(self)
    return Variable.from_numpy(arr, False)


def _grad_var(self: Variable) -> Optional[Variable]:
    try:
        # If backend defines a grad method, call it directly (class-level) to
        # avoid hitting any wrapper property that might have been attached.
        backend_grad = getattr(_backend.Variable, "grad", None)
        if callable(backend_grad):
            g = backend_grad(self)
        else:
            # Try instance-level call if class-level missing
            g = getattr(self, "grad", None)
            if callable(g):
                g = g()
            else:
                g = None
    except Exception:
        g = None
    if g is None:
        return None
    arr = np.asarray(g, dtype=np.float32)
    if arr.size == 0:
        return None
    try:
        arr = arr.reshape(tuple(self.shape()))
    except Exception:
        pass
    return Variable.from_numpy(arr, False)

try:
    # Only set .value and .grad properties if the backend doesn't already
    # provide them. Overwriting backend methods previously caused recursion.
    if not hasattr(_backend.Variable, "value"):
        Variable.value = property(lambda self: _value_var(self))
    if not hasattr(_backend.Variable, "grad"):
        Variable.grad = property(lambda self: _grad_var(self))
except Exception:
    pass

def value_var(v: Variable) -> Variable:
    return _value_var(v)


def grad_var(v: Variable) -> Optional[Variable]:
    return _grad_var(v)

Variable.__array_priority__ = 1000

Variable.__add__ = _binary(add)
Variable.__radd__ = _rbinary(add)

Variable.__sub__ = _binary(sub)
Variable.__rsub__ = _rbinary(sub)

Variable.__mul__ = _binary(mul)
Variable.__rmul__ = _rbinary(mul)

Variable.__truediv__ = _binary(div)
Variable.__rtruediv__ = _rbinary(div)

Variable.__neg__ = lambda self: neg(self)

Variable.__pow__ = _binary(pow)
Variable.__rpow__ = _rbinary(pow)

Variable.__rmatmul__ = _rbinary(matmul)

def _t(self: Variable) -> Variable:
    return transpose(self)
Variable.t = _t

Variable.numpy = _numpy

def _wrap_binary_scalar(op_backend, a, b):
    a_is_var = isinstance(a, Variable)
    b_is_var = isinstance(b, Variable)
    if a_is_var and b_is_var:
        return op_backend(a, b)
    if a_is_var and not b_is_var:
        other = _ensure_var(b, like=a, requires_grad=a.requires_grad())
        return op_backend(a, other)
    if b_is_var and not a_is_var:
        other = _ensure_var(a, like=b, requires_grad=b.requires_grad())
        return op_backend(other, b)
    va = _ensure_var(a, requires_grad=False)
    vb = _ensure_var(b, requires_grad=False)
    return op_backend(va, vb)

def add(a, b):
    return _wrap_binary_scalar(_backend.add, a, b)

def sub(a, b):
    return _wrap_binary_scalar(_backend.sub, a, b)

def mul(a, b):
    return _wrap_binary_scalar(_backend.mul, a, b)

Variable.__add__ = _binary(add)
Variable.__radd__ = _rbinary(add)
Variable.__sub__ = _binary(sub)
Variable.__rsub__ = _rbinary(sub)
Variable.__mul__ = _binary(mul)
Variable.__rmul__ = _rbinary(mul)

Variable.__array_priority__ = 1000

__all__ = [
    "Variable",
    "tensor", "zeros", "ones", "full", "randn",
    "add", "sub", "mul", "div", "neg", "pow",
    "sin", "cos", "exp", "log",
    "relu", "sigmoid", "tanh", "clamp",
    "matmul", "transpose", "flatten", "reshape",
    "reduce_sum", "reduce_mean", "broadcast_to",
    "is_grad_enabled", "set_grad_enabled", "nograd", "stop_gradient", "detach",
]

if nn is not None:
    __all__.append("nn")
