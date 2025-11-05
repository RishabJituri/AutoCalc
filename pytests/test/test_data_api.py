import numpy as np
import ag
import pytest


def to_numpy(obj):
    """Robust conversion of backend objects to numpy arrays for tests."""
    try:
        if callable(getattr(obj, "value", None)):
            return np.asarray(obj.value(), dtype=np.float32)
        v = getattr(obj, "value", None)
        if v is not None:
            return np.asarray(v, dtype=np.float32)
    except Exception:
        pass
    try:
        if callable(getattr(obj, "numpy", None)):
            return np.asarray(obj.numpy(), dtype=np.float32)
    except Exception:
        pass
    try:
        return np.asarray(obj, dtype=np.float32)
    except Exception:
        return None


class RandDataset(ag.data.Dataset):
    """Small Python Dataset producing deterministic samples for tests."""
    def __init__(self, n, x_shape=(1, 4, 4)):
        super().__init__()
        self._n = n
        self._x_shape = x_shape
    def size(self):
        return self._n
    def get(self, idx):
        x = np.full(self._x_shape, float(idx), dtype=np.float32)
        y = np.array([idx % 10], dtype=np.float32)
        ex = ag.data.Example()
        ex.x = ag.Variable.from_numpy(x, requires_grad=False)
        ex.y = ag.Variable.from_numpy(y, requires_grad=False)
        return ex


def test_dataloader_iteration_and_batch_sizes():
    N = 23
    batch_size = 6
    ds = RandDataset(N)

    opts = ag.data.DataLoaderOptions()
    opts.batch_size = batch_size
    opts.shuffle = False

    loader = ag.data.DataLoader(ds, opts)

    # iterate and count batches, inspect batch sizes and contents
    counts = []
    idxs = []
    for i, batch in enumerate(loader):
        # batch.size field reports the effective batch size
        assert hasattr(batch, 'size')
        b = int(batch.size)
        counts.append(b)
        arr = to_numpy(batch.x)
        assert arr is not None
        # first element of this batch should equal the minimum index value used in this batch
        idxs.append(int(arr.reshape(b, -1)[0, 0]))

    # Expect ceil(N / batch_size) batches
    expected_batches = (N + batch_size - 1) // batch_size
    assert len(counts) == expected_batches
    # Last batch size should be N % batch_size (if not zero)
    if N % batch_size == 0:
        assert counts[-1] == batch_size
    else:
        assert counts[-1] == (N % batch_size)


def test_dataloader_shuffle_and_indices_rewind():
    N = 16
    ds = RandDataset(N)
    opts = ag.data.DataLoaderOptions()
    opts.batch_size = 4
    opts.shuffle = True
    opts.seed = 12345

    loader = ag.data.DataLoader(ds, opts)
    inds1 = list(loader.indices())
    # rewind (increment epoch) and get new ordering
    loader.rewind()
    inds2 = list(loader.indices())
    # When shuffle=True and epoch changes, ordering should change for N>1
    if N > 1:
        assert inds1 != inds2
    # indices length should remain N
    assert len(inds1) == N and len(inds2) == N


def test_transformdataset_applies_function():
    N = 10
    ds = RandDataset(N)

    def add_one(example):
        # mutate x by adding 1.0
        x_np = to_numpy(example.x)
        x_np = x_np + 1.0
        out = ag.data.Example()
        out.x = ag.Variable.from_numpy(x_np, requires_grad=False)
        out.y = example.y
        return out

    td = ag.data.TransformDataset(ds, add_one)
    # get an element from transform dataset and check transform applied
    ex0 = td.get(3)
    arr = to_numpy(ex0.x)
    assert arr is not None
    # original RandDataset would have filled array with 3.0
    assert np.allclose(arr, 4.0)
