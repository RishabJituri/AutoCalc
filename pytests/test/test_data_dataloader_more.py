import numpy as np
import ag


def to_numpy_var(v):
    try:
        if callable(getattr(v, "value", None)):
            return np.asarray(v.value())
    except Exception:
        pass
    try:
        return np.asarray(v)
    except Exception:
        return None


class RandDataset(ag.data.Dataset):
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


def test_drop_last_and_len():
    N = 10
    bs = 4
    ds = RandDataset(N)
    opts = ag.data.DataLoaderOptions()
    opts.batch_size = bs
    opts.drop_last = True

    loader = ag.data.DataLoader(ds, opts)

    # expected full batches = floor(N / bs)
    expected = N // bs
    # __len__ binding should report number of batches
    assert int(loader.__len__()) == expected

    cnt = 0
    for batch in loader:
        assert int(batch.size) == bs
        cnt += 1
    assert cnt == expected


def test_iterator_reset_vs_rewind_seeded_shuffle():
    N = 20
    ds = RandDataset(N)
    opts = ag.data.DataLoaderOptions()
    opts.batch_size = 5
    opts.shuffle = True
    opts.seed = 12345

    loader = ag.data.DataLoader(ds, opts)
    inds_before = list(loader.indices())

    # Iterating calls reset() via __iter__, but does not increment epoch, so ordering should stay same
    _ = list(loader)  # consume
    inds_after_consume = list(loader.indices())
    assert inds_before == inds_after_consume

    # rewind() increments epoch -> ordering should change (for N>1)
    loader.rewind()
    inds_after_rewind = list(loader.indices())
    if N > 1:
        assert inds_before != inds_after_rewind
    assert len(inds_after_rewind) == N


def test_compose_and_transformdataset():
    N = 6
    ds = RandDataset(N)

    # Compose two transforms: add 1, then multiply by 2 (on x)
    def add_one(ex):
        o = ag.data.Example()
        x = to_numpy_var(ex.x)
        o.x = ag.Variable.from_numpy(x + 1.0, requires_grad=False)
        o.y = ex.y
        return o

    def times_two(ex):
        o = ag.data.Example()
        x = to_numpy_var(ex.x)
        o.x = ag.Variable.from_numpy(x * 2.0, requires_grad=False)
        o.y = ex.y
        return o

    c = ag.data.Compose()
    c.ts = [add_one, times_two]

    td = ag.data.TransformDataset(ds, c)
    e = td.get(3)
    arr = to_numpy_var(e.x)
    # original value 3 -> (3 + 1) * 2 = 8
    assert np.allclose(arr, 8.0)
