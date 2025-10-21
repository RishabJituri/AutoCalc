import numpy as np
import ag


def to_numpy(var):
    vals = np.array(var.value(), dtype=np.float32)
    return vals.reshape(tuple(var.shape()))


def grad_to_numpy(var):
    g = np.array(var.grad(), dtype=np.float32)
    return g.reshape(tuple(var.shape()))


def test_slice_forward_and_backward():
    # Create a 3D tensor: shape (2,3,4)
    a = np.arange(2*3*4, dtype=np.float32).reshape(2,3,4)
    A = ag.Variable.from_numpy(a, True)

    # Slice: [:,1:3,2:4] -> shape (2,2,2)
    S = A.__getitem__((slice(None), slice(1,3), slice(2,4)))

    # Forward: compare to numpy
    expected = a[:,1:3,2:4]
    got = to_numpy(S)
    assert np.allclose(got, expected)

    # Backward: sum of slice -> gradients should be ones in sliced region
    loss = ag.reduce_sum(S)
    loss.backward()

    gA = grad_to_numpy(A)
    expected_grad = np.zeros_like(a)
    expected_grad[:,1:3,2:4] = 1.0
    assert np.allclose(gA, expected_grad)


def test_chained_indexing_and_integer_index():
    a = np.arange(2*3*4, dtype=np.float32).reshape(2,3,4)
    A = ag.Variable.from_numpy(a, True)

    # Chained indexing: [:][0][0] equivalent to a[0,0,:]
    v0 = A.__getitem__(slice(None))
    v1 = v0.__getitem__(0)  # picks first of second dim
    v2 = v1.__getitem__(0)  # picks first of third dim -> scalar vector

    expected = a[0,0,:]
    got = to_numpy(v2)
    assert np.allclose(got, expected)

    # Test backward for integer-index (should map to single element)
    A.zero_grad()
    s = ag.reduce_sum(v2)  # sum of selected vector
    s.backward()
    gA = grad_to_numpy(A)
    expected_grad = np.zeros_like(a)
    expected_grad[0,0,:] = 1.0
    assert np.allclose(gA, expected_grad)
