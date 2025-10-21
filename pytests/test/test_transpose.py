import numpy as np
import ag


def to_numpy(var):
    vals = np.array(var.value(), dtype=np.float32)
    return vals.reshape(tuple(var.shape()))


def grad_to_numpy(var):
    g = np.array(var.grad(), dtype=np.float32)
    return g.reshape(tuple(var.shape()))


def test_transpose_forward_and_matmul():
    # A: (3,2), B: (3,4) -> C = A.T @ B  where A.T is (2,3) -> result (2,4)
    a = np.arange(3*2, dtype=np.float32).reshape(3,2)
    b = np.arange(3*4, dtype=np.float32).reshape(3,4).astype(np.float32)
    A = ag.Variable.from_numpy(a, True)
    B = ag.Variable.from_numpy(b, True)

    At = A.T  # materialized transpose
    C = ag.matmul(At, B)

    expected = a.T @ b
    got = to_numpy(C)
    assert np.allclose(got, expected)

    # Backward: sum(C) -> grads accumulate into A and B appropriately
    C.zero_grad(); A.zero_grad(); B.zero_grad()
    s = ag.reduce_sum(C)
    s.backward()

    gA = grad_to_numpy(A)
    gB = grad_to_numpy(B)
    assert gA.shape == a.shape
    assert gB.shape == b.shape
    assert np.any(gA != 0)
    assert np.any(gB != 0)


def test_transpose_batched_and_grad():
    # Batched: A: (2,3,2) B: (2,3,4) -> C: (2,2,4) for each batch, using A.T per batch
    a = np.arange(2*3*2, dtype=np.float32).reshape(2,3,2)
    b = np.arange(2*3*4, dtype=np.float32).reshape(2,3,4).astype(np.float32)
    A = ag.Variable.from_numpy(a, True)
    B = ag.Variable.from_numpy(b, True)

    At = A.T
    C = ag.matmul(At, B)

    expected = np.zeros((2,2,4), dtype=np.float32)
    for i in range(2): expected[i] = a[i].T @ b[i]
    got = to_numpy(C)
    assert np.allclose(got, expected)

    C.zero_grad(); A.zero_grad(); B.zero_grad()
    s = ag.reduce_sum(C)
    s.backward()
    gA = grad_to_numpy(A)
    gB = grad_to_numpy(B)
    assert gA.shape == a.shape
    assert gB.shape == b.shape
    assert np.any(gA != 0)
    assert np.any(gB != 0)
