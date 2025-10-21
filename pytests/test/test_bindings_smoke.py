import numpy as np
import ag


def to_numpy(var):
    vals = np.array(var.value(), dtype=np.float32)
    return vals.reshape(tuple(var.shape()))


def grad_to_numpy(var):
    g = np.array(var.grad(), dtype=np.float32)
    if g.size == 0:
        return g
    return g.reshape(tuple(var.shape()))


def test_add_mul_grad():
    a = np.arange(2*3, dtype=np.float32).reshape(2,3)
    b = np.ones_like(a)
    A = ag.Variable.from_numpy(a, True)
    B = ag.Variable.from_numpy(b, True)

    x = ag.add(A, B)
    y = ag.mul(A, B)
    loss = ag.reduce_sum(x) + ag.reduce_sum(y)
    loss.backward()

    gA = grad_to_numpy(A)
    gB = grad_to_numpy(B)

    # d/dA of sum(A+B) + sum(A*B) = 1 + B
    expected_gA = 1.0 + b
    expected_gB = 1.0 + a
    assert np.allclose(gA, expected_gA)
    assert np.allclose(gB, expected_gB)


def test_matmul_batched_and_grad():
    a = np.arange(2*3*4, dtype=np.float32).reshape(2,3,4)
    b = np.arange(2*4*5, dtype=np.float32).reshape(2,4,5).astype(np.float32)
    A = ag.Variable.from_numpy(a, True)
    B = ag.Variable.from_numpy(b, True)

    C = ag.matmul(A, B)
    expected = np.zeros((2,3,5), dtype=np.float32)
    for i in range(2): expected[i] = a[i] @ b[i]
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


def test_reshape_flatten_broadcast():
    a = np.arange(2*3*4, dtype=np.float32).reshape(2,3,4)
    A = ag.Variable.from_numpy(a, True)

    R = ag.reshape(A, [6,4])
    assert tuple(R.shape()) == (6,4)
    F = ag.flatten(A, 1)
    assert tuple(F.shape()) == (2,12)

    # broadcast
    B = ag.broadcast_to(ag.Variable.from_numpy(np.array([1.0], dtype=np.float32), False), [2,3,4])
    assert tuple(B.shape()) == (2,3,4)


def test_relu_and_ops_smoke():
    a = np.linspace(-1.0, 1.0, 6, dtype=np.float32).reshape(2,3)
    A = ag.Variable.from_numpy(a, True)
    R = ag.relu(A)
    S = ag.sigmoid(A)
    E = ag.exp(A)
    P = ag.pow(A, ag.Variable.from_numpy(np.full(a.shape, 2.0, dtype=np.float32), False))

    loss = ag.reduce_sum(R) + ag.reduce_sum(S) + ag.reduce_sum(E) + ag.reduce_sum(P)
    loss.backward()
    gA = grad_to_numpy(A)
    assert gA.shape == a.shape
    assert np.any(gA != 0)


def test_stop_gradient_and_detach():
    a = np.arange(6, dtype=np.float32).reshape(2,3)
    A = ag.Variable.from_numpy(a, True)

    X = ag.stop_gradient(A)
    s = ag.reduce_sum(X)
    s.backward()
    # stop_gradient should prevent grad allocation on original
    assert len(A.grad()) == 0

    # detach should behave similarly
    A.zero_grad()
    Y = ag.detach(A)
    s2 = ag.reduce_sum(Y)
    s2.backward()
    assert len(A.grad()) == 0


def test_transpose_and_getitem_smoke():
    a = np.arange(2*3, dtype=np.float32).reshape(2,3)
    A = ag.Variable.from_numpy(a, True)
    At = A.T
    assert tuple(At.shape()) == (3,2)

    # getitem slice
    S = A.__getitem__((slice(None), slice(1,3)))
    assert tuple(S.shape()) == (2,2)

    # integer indexing reduces rank
    v = A.__getitem__(0)
    assert len(v.shape()) == 1

