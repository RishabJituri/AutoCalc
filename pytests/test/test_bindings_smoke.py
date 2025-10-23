import numpy as np
import pytest
import ag

# ---- helpers ----
def to_numpy(var):
    # Prefer a Variable.numpy() method if available, otherwise fall back to calling .value() or using numpy.asarray
    try:
        if hasattr(var, 'numpy') and callable(getattr(var, 'numpy')):
            return np.asarray(var.numpy(), dtype=np.float32)
        # if .value is a method, call it
        if hasattr(var, 'value') and callable(getattr(var, 'value')):
            vals = np.array(var.value(), dtype=np.float32)
            try:
                return vals.reshape(tuple(var.shape()))
            except Exception:
                return vals
        # fallback
        vals = np.asarray(var, dtype=np.float32)
        return vals
    except Exception:
        return np.asarray(var, dtype=np.float32)

def grad_to_numpy(var):
    # grad() may be a method or attribute; handle both.
    try:
        g = var.grad() if callable(getattr(var, 'grad', None)) else getattr(var, 'grad', None)
    except Exception:
        g = getattr(var, 'grad', None)
    if g is None:
        return np.array([], dtype=np.float32)
    arr = np.array(g, dtype=np.float32)
    if arr.size == 0:
        return arr
    try:
        return arr.reshape(tuple(var.shape()))
    except Exception:
        return arr

# ---- basic creation / shape ----
def test_create_from_numpy_and_shape():
    a = np.arange(6, dtype=np.float32).reshape(2,3)
    A = ag.Variable.from_numpy(a, True)
    assert tuple(A.shape()) == (2,3)
    np.testing.assert_allclose(to_numpy(A), a)

def test_dunder_ops_elementwise_and_broadcast_scalar():
    a = np.arange(6, dtype=np.float32).reshape(2,3)
    b = np.ones_like(a, dtype=np.float32) * 2.0

    A = ag.Variable.from_numpy(a, False)
    B = ag.Variable.from_numpy(b, False)

    # +, -, *, / with scalar broadcasting
    np.testing.assert_allclose(to_numpy(A + 2.0), a + 2.0)
    np.testing.assert_allclose(to_numpy(2.0 + A), 2.0 + a)

    np.testing.assert_allclose(to_numpy(A - 2.0), a - 2.0)
    np.testing.assert_allclose(to_numpy(2.0 - A), 2.0 - a)

    np.testing.assert_allclose(to_numpy(A * 2.0), a * 2.0)
    np.testing.assert_allclose(to_numpy(2.0 * A), 2.0 * a)

    np.testing.assert_allclose(to_numpy(A / 2.0), a / 2.0)
    np.testing.assert_allclose(to_numpy(2.0 / A), 2.0 / a)

    # elementwise with tensors
    np.testing.assert_allclose(to_numpy(A + B), a + b)
    np.testing.assert_allclose(to_numpy(A * B), a * b)

def test_matmul_and_transpose_shapes_and_values():
    a = np.arange(6, dtype=np.float32).reshape(2,3)
    A = ag.Variable.from_numpy(a, False)
    # Use .T property (bound in C++) and also ag.transpose
    At = A.T
    np.testing.assert_allclose(to_numpy(At), a.T)

    out = A @ At
    np.testing.assert_allclose(to_numpy(out), a @ a.T)
    assert tuple(out.shape()) == (2,2)

    out2 = ag.matmul(A, ag.transpose(A))
    np.testing.assert_allclose(to_numpy(out2), a @ a.T)

def test_reduce_and_backward_simple_dot():
    # L = sum(A * B) => dL/dA = B ; dL/dB = A
    a = np.arange(6, dtype=np.float32).reshape(2,3)
    b = np.linspace(1, 2, 6, dtype=np.float32).reshape(2,3)
    A = ag.Variable.from_numpy(a, True)
    B = ag.Variable.from_numpy(b, True)

    dot = A * B
    loss = ag.reduce_sum(dot)
    loss.backward()

    np.testing.assert_allclose(grad_to_numpy(A), b)
    np.testing.assert_allclose(grad_to_numpy(B), a)

def test_unary_and_activations_exist_and_run():
    a = np.linspace(-2, 2, 6, dtype=np.float32).reshape(2,3)
    A = ag.Variable.from_numpy(a, False)

    np.testing.assert_allclose(to_numpy(ag.neg(A)), -a)
    np.testing.assert_allclose(to_numpy(ag.sin(A)), np.sin(a), rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(to_numpy(ag.cos(A)), np.cos(a), rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(to_numpy(ag.exp(A)), np.exp(a), rtol=1e-5, atol=1e-6)
    # clamp then relu equivalence for positive part
    C = ag.clamp(A, 0.0, 999.0)
    R = ag.relu(A)
    np.testing.assert_allclose(to_numpy(C), np.maximum(a, 0.0))
    np.testing.assert_allclose(to_numpy(R), np.maximum(a, 0.0))

    # sigmoid/tanh sanity (range)
    S = to_numpy(ag.sigmoid(A))
    T = to_numpy(ag.tanh(A))
    assert np.all(S > 0) and np.all(S < 1)
    assert np.all(T >= -1) and np.all(T <= 1)

def test_broadcast_reshape_flatten_and_indexing():
    a = np.arange(6, dtype=np.float32).reshape(2,3)
    A = ag.Variable.from_numpy(a, False)

    # broadcast_to
    B = ag.broadcast_to(A, (2,3,3))
    assert tuple(B.shape()) == (2,3,3)
    expected = to_numpy(ag.broadcast_to(ag.Variable.from_numpy(a), (2,3,3)))
    np.testing.assert_allclose(to_numpy(B), expected)

    # reshape
    R = ag.reshape(A, (3,2))
    assert tuple(R.shape()) == (3,2)
    np.testing.assert_allclose(to_numpy(R), a.reshape(3,2))

    # flatten (start_dim=1) turns (2,3) -> (2,3) here; start_dim=0 -> (6,)
    F = ag.flatten(A, start_dim=0)
    assert tuple(F.shape()) == (6,)
    np.testing.assert_allclose(to_numpy(F), a.reshape(6,))

    # __getitem__ slice and int
    S = A.__getitem__((slice(None), slice(1,3)))
    assert tuple(S.shape()) == (2,2)
    np.testing.assert_allclose(to_numpy(S), a[:,1:3])

    v = A.__getitem__(0)
    assert len(v.shape()) == 1
    np.testing.assert_allclose(to_numpy(v), a[0])

def test_grad_mode_context_manager():
    # Ensure is_grad_enabled / set_grad_enabled / nograd work together
    st0 = ag.is_grad_enabled()
    try:
        ag.set_grad_enabled(True)
        assert ag.is_grad_enabled() is True
        with ag.nograd():
            assert ag.is_grad_enabled() is False
        assert ag.is_grad_enabled() is True
    finally:
        ag.set_grad_enabled(st0)

def test_broadcast_edge_cases_and_backward_accumulation():
    # right/left alignment behaviour and backward accumulation
    a = np.arange(6, dtype=np.float32).reshape(2,3)
    A = ag.Variable.from_numpy(a, True)

    # left-aligned broadcast (extra trailing dimension) should work
    B = ag.broadcast_to(A, (2,3,3))
    assert tuple(B.shape()) == (2,3,3)
    expected = to_numpy(ag.broadcast_to(ag.Variable.from_numpy(a), (2,3,3)))
    np.testing.assert_allclose(to_numpy(B), expected)

    # ensure other broadcast shapes also work (right-aligned expansion)
    B2 = ag.broadcast_to(A, (3,2,3))
    assert tuple(B2.shape()) == (3,2,3)
    expected2 = to_numpy(ag.broadcast_to(ag.Variable.from_numpy(a), (3,2,3)))
    np.testing.assert_allclose(to_numpy(B2), expected2)

    # backward accumulation: broadcasting (2,1) -> (2,3) sums correctly
    x = np.array([[1.0],[2.0]], dtype=np.float32)
    X = ag.Variable.from_numpy(x, True)
    Y = ag.broadcast_to(X, (2,3))
    loss = ag.reduce_sum(Y)
    loss.backward()
    # each input element is replicated 3 times along last axis, so grad should be 3 for each
    np.testing.assert_allclose(grad_to_numpy(X), np.array([[3.0],[3.0]], dtype=np.float32))


def test_integer_index_rank_reduction_multiple_dims():
    a = np.arange(24, dtype=np.float32).reshape(2,3,4)
    A = ag.Variable.from_numpy(a, False)
    v = A.__getitem__((0, 1))  # should reduce first two dims -> shape (4,)
    assert len(v.shape()) == 1
    np.testing.assert_allclose(to_numpy(v), a[0,1])


def test_reverse_ops_with_numpy_array_left_operand():
    a = np.arange(6, dtype=np.float32).reshape(2,3)
    A = ag.Variable.from_numpy(a, False)
    arr = np.ones_like(a, dtype=np.float32) * 2.0
    # numpy should dispatch to our Variable.__radd__ / __rmul__ because of __array_priority__
    np.testing.assert_allclose(to_numpy(arr + A), arr + a)
    np.testing.assert_allclose(to_numpy(arr * A), arr * a)


def test_flatten_and_reshape_backward_accumulates():
    a = np.arange(6, dtype=np.float32).reshape(2,3)
    A = ag.Variable.from_numpy(a, True)

    # flatten backward
    F = ag.flatten(A, start_dim=0)
    loss = ag.reduce_sum(F)
    loss.backward()
    np.testing.assert_allclose(grad_to_numpy(A), np.ones_like(a, dtype=np.float32))

    # reshape backward: recreate A fresh and test
    A = ag.Variable.from_numpy(a, True)
    R = ag.reshape(A, (3,2))
    loss = ag.reduce_sum(R)
    loss.backward()
    np.testing.assert_allclose(grad_to_numpy(A), np.ones_like(a, dtype=np.float32))
