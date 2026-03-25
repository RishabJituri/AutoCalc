# filepath: pytests/test/test_milestone1_ops.py
"""
Tests for Milestone 1 new ops: cat, upsample2d, softmax, argmax_lastdim,
reduce_max, logsumexp, and Sequential Python construction.
"""
import numpy as np
import pytest
import ag


# ===== cat =====

class TestCat:
    def test_forward_axis0(self):
        a = ag.tensor(np.array([[1, 2], [3, 4]], dtype=np.float32))
        b = ag.tensor(np.array([[5, 6]], dtype=np.float32))
        c = ag.cat([a, b], axis=0)
        out = np.asarray(c.value()).reshape(c.shape())
        expected = np.array([[1, 2], [3, 4], [5, 6]], dtype=np.float32)
        np.testing.assert_allclose(out, expected, atol=1e-6)

    def test_forward_axis1(self):
        a = ag.tensor(np.array([[1, 2], [3, 4]], dtype=np.float32))
        b = ag.tensor(np.array([[5], [6]], dtype=np.float32))
        c = ag.cat([a, b], axis=1)
        out = np.asarray(c.value()).reshape(c.shape())
        expected = np.array([[1, 2, 5], [3, 4, 6]], dtype=np.float32)
        np.testing.assert_allclose(out, expected, atol=1e-6)

    def test_backward_axis0(self):
        a = ag.Variable.from_numpy(np.array([[1, 2], [3, 4]], dtype=np.float32), True)
        b = ag.Variable.from_numpy(np.array([[5, 6]], dtype=np.float32), True)
        c = ag.cat([a, b], axis=0)
        s = ag.reduce_sum(c)
        s.backward()
        ga = np.asarray(a.grad()).reshape(a.shape())
        gb = np.asarray(b.grad()).reshape(b.shape())
        np.testing.assert_allclose(ga, np.ones((2, 2), dtype=np.float32), atol=1e-6)
        np.testing.assert_allclose(gb, np.ones((1, 2), dtype=np.float32), atol=1e-6)

    def test_three_inputs(self):
        a = ag.tensor(np.ones((1, 3), dtype=np.float32))
        b = ag.tensor(np.ones((2, 3), dtype=np.float32) * 2)
        c = ag.tensor(np.ones((3, 3), dtype=np.float32) * 3)
        d = ag.cat([a, b, c], axis=0)
        assert tuple(d.shape()) == (6, 3)


# ===== upsample2d =====

class TestUpsample2d:
    def test_forward_2x(self):
        x_np = np.array([[[[1, 2], [3, 4]]]], dtype=np.float32)  # [1,1,2,2]
        x = ag.tensor(x_np)
        y = ag.upsample2d(x, 2, 2)
        assert tuple(y.shape()) == (1, 1, 4, 4)
        out = np.asarray(y.value()).reshape(y.shape())
        expected = np.array([
            [1, 1, 2, 2],
            [1, 1, 2, 2],
            [3, 3, 4, 4],
            [3, 3, 4, 4],
        ], dtype=np.float32).reshape(1, 1, 4, 4)
        np.testing.assert_allclose(out, expected, atol=1e-6)

    def test_backward_accumulates(self):
        x = ag.Variable.from_numpy(
            np.array([[[[1, 2], [3, 4]]]], dtype=np.float32), True
        )
        y = ag.upsample2d(x, 2, 2)
        s = ag.reduce_sum(y)
        s.backward()
        gx = np.asarray(x.grad()).reshape(x.shape())
        # Each pixel replicated 4 times -> grad = 4
        np.testing.assert_allclose(gx, np.full((1, 1, 2, 2), 4.0, dtype=np.float32), atol=1e-6)

    def test_multi_batch_channel(self):
        x_np = np.arange(1, 7, dtype=np.float32).reshape(2, 3, 1, 1)
        x = ag.tensor(x_np)
        y = ag.upsample2d(x, 3, 3)
        assert tuple(y.shape()) == (2, 3, 3, 3)


# ===== softmax =====

class TestSoftmax:
    def test_basic(self):
        x_np = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
        x = ag.tensor(x_np)
        y = ag.softmax(x, axis=-1)
        out = np.asarray(y.value()).reshape(y.shape())
        expected = np.exp(x_np) / np.sum(np.exp(x_np), axis=-1, keepdims=True)
        np.testing.assert_allclose(out, expected, atol=1e-5)

    def test_sums_to_one(self):
        x = ag.tensor(np.random.randn(4, 10).astype(np.float32))
        y = ag.softmax(x, axis=-1)
        out = np.asarray(y.value()).reshape(y.shape())
        row_sums = out.sum(axis=-1)
        np.testing.assert_allclose(row_sums, np.ones(4), atol=1e-5)


# ===== argmax_lastdim =====

class TestArgmax:
    def test_basic(self):
        x_np = np.array([[1, 5, 3], [7, 2, 4]], dtype=np.float32)
        x = ag.tensor(x_np)
        indices = ag.argmax_lastdim(x)
        assert list(indices) == [1, 0]

    def test_single_row(self):
        x = ag.tensor(np.array([[10, 20, 5]], dtype=np.float32))
        indices = ag.argmax_lastdim(x)
        assert list(indices) == [1]


# ===== reduce_max =====

class TestReduceMax:
    def test_all_axes(self):
        x_np = np.array([[1, 5], [3, 2]], dtype=np.float32)
        x = ag.tensor(x_np)
        y = ag.reduce_max(x)
        out = np.asarray(y.value()).flat[0]
        assert abs(out - 5.0) < 1e-6

    def test_keepdims(self):
        x_np = np.array([[1, 5], [3, 2]], dtype=np.float32)
        x = ag.tensor(x_np)
        y = ag.reduce_max(x, axes=[1], keepdims=True)
        assert tuple(y.shape()) == (2, 1)


# ===== logsumexp =====

class TestLogsumexp:
    def test_basic(self):
        x_np = np.array([[1, 2, 3]], dtype=np.float32)
        x = ag.tensor(x_np)
        y = ag.logsumexp(x, axes=[1])
        out = np.asarray(y.value()).flat[0]
        expected = np.log(np.sum(np.exp(x_np)))
        assert abs(out - expected) < 1e-4


# ===== Sequential constructor from Python =====

class TestSequentialPython:
    def test_construct_and_forward(self):
        seq = ag.nn.Sequential()
        seq.append(ag.nn.Linear(4, 8))
        seq.append(ag.nn.Linear(8, 2))
        x = ag.tensor(np.random.randn(1, 4).astype(np.float32))
        y = seq(x)
        assert tuple(y.shape()) == (1, 2)

    def test_parameters_collected(self):
        seq = ag.nn.Sequential()
        seq.append(ag.nn.Linear(4, 8, bias=True))
        seq.append(ag.nn.Linear(8, 2, bias=True))
        params = seq.parameters()
        # Two Linear layers with weight+bias each = 4 params
        assert len(params) >= 4
