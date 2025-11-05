import numpy as np
import ag


def to_numpy_var(v):
    # try common access patterns used in the bindings/tests
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
        x = np.ones(self._x_shape, dtype=np.float32) * float(idx)
        y = np.array([idx % 10], dtype=np.float32)
        ex = ag.data.Example()
        ex.x = ag.Variable.from_numpy(x, requires_grad=False)
        ex.y = ag.Variable.from_numpy(y, requires_grad=False)
        return ex


def test_python_dataset_iteration_and_batching():
    N = 37
    bs = 8
    ds = RandDataset(N)

    opts = ag.data.DataLoaderOptions()
    opts.batch_size = bs
    opts.shuffle = False

    loader = ag.data.DataLoader(ds, opts)

    # iterate and count batches, verify batch sizes and ordering when shuffle=False
    seen = 0
    idx = 0
    for batch in loader:
        assert hasattr(batch, 'x') and hasattr(batch, 'y')
        shp = batch.x.shape()
        # first dim should be batch size except possibly last one
        if batch.size < bs:
            assert batch.size == N % bs
        else:
            assert batch.size == bs
        # check that the first element in the batch corresponds to the expected index
        arr = to_numpy_var(batch.x)
        assert arr is not None
        # values were filled with the sample idx, so first element's entry should equal idx
        first_val = float(arr[0].sum()) / float(arr[0].size)
        # since x was ones * sample_idx, average equals idx
        assert abs(first_val - float(idx)) < 1e-6
        seen += 1
        idx += batch.size

    assert idx == N
    # number of batches should equal ceil(N/bs)
    import math
    assert seen == math.ceil(N / bs)
