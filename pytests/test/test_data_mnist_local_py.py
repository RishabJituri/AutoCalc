import os
import numpy as np
import ag
import pytest
import struct


def var_to_numpy(v):
    try:
        if callable(getattr(v, "value", None)):
            return np.asarray(v.value())
        v_raw = getattr(v, "value", None)
        if v_raw is not None:
            return np.asarray(v_raw)
    except Exception:
        pass
    try:
        if callable(getattr(v, "numpy", None)):
            return np.asarray(v.numpy())
    except Exception:
        pass
    try:
        return np.asarray(v)
    except Exception:
        return None


class LocalMNIST(ag.data.Dataset):
    """Read local MNIST idx files under Data/MNIST/raw without torchvision."""
    def __init__(self, root, train=True, max_samples=None):
        super().__init__()
        
        mnist_raw_dir = os.path.join(root, "MNIST", "raw")
        if train:
            img_path = os.path.join(mnist_raw_dir, "train-images-idx3-ubyte")
            lbl_path = os.path.join(mnist_raw_dir, "train-labels-idx1-ubyte")
        else:
            img_path = os.path.join(mnist_raw_dir, "t10k-images-idx3-ubyte")
            lbl_path = os.path.join(mnist_raw_dir, "t10k-labels-idx1-ubyte")
        if not (os.path.exists(img_path) and os.path.exists(lbl_path)):
            raise RuntimeError(f"MNIST idx files not found under {mnist_raw_dir}")

        # load images
        with open(img_path, 'rb') as f:
            magic, num, rows, cols = struct.unpack('>IIII', f.read(16))
            data = np.frombuffer(f.read(), dtype=np.uint8)
            data = data.reshape(num, rows, cols)
        # load labels
        with open(lbl_path, 'rb') as f:
            magic_l, num_l = struct.unpack('>II', f.read(8))
            labels = np.frombuffer(f.read(), dtype=np.uint8)

        if max_samples:
            num_take = min(max_samples, num)
            self._images = data[:num_take]
            self._labels = labels[:num_take]
        else:
            self._images = data
            self._labels = labels
        self._max = max_samples

    def size(self):
        return int(self._images.shape[0])

    def get(self, idx):
        arr = self._images[int(idx)].astype(np.float32) / 255.0
        if arr.ndim == 2:
            arr = arr[None, :, :]
        ex = ag.data.Example()
        ex.x = ag.Variable.from_numpy(arr, requires_grad=False)
        ex.y = ag.Variable.from_numpy(np.array([int(self._labels[int(idx)])], dtype=np.int64), requires_grad=False)
        return ex


@pytest.mark.slow
def test_mnist_local_dataloader_basic():
    # Expect local MNIST files under ./Data
    data_root = os.path.abspath(os.path.join(os.getcwd(), "Data"))
    mnist_raw = os.path.join(data_root, "MNIST", "raw", "train-images-idx3-ubyte")
    if not os.path.exists(mnist_raw):
        pytest.skip(f"Local MNIST not found under {data_root}; skipping")

    N = 128
    bs = 32
    ds = LocalMNIST(data_root, train=True, max_samples=N)

    opts = ag.data.DataLoaderOptions()
    opts.batch_size = bs
    opts.shuffle = False
    loader = ag.data.DataLoader(ds, opts)

    inds = list(loader.indices())
    assert len(inds) == N

    n_batches = 0
    total_seen = 0
    for batch in loader:
        n = int(batch.size)
        assert 0 < n <= bs
        x_np = var_to_numpy(batch.x)
        y_np = var_to_numpy(batch.y)
        assert x_np is not None and y_np is not None
        # Some collate implementations may return flattened batch arrays.
        # If flattened (1D) and matches N * 28*28, reshape to (N,1,28,28).
        if x_np.ndim == 1 and x_np.size == n * 28 * 28:
            x_np = x_np.reshape(n, 1, 28, 28)
        elif x_np.ndim == 2 and x_np.shape[0] == n and x_np.shape[1] == 28 * 28:
            x_np = x_np.reshape(n, 1, 28, 28)
        assert x_np.ndim == 4 and x_np.shape[0] == n
        assert y_np.shape[0] == n
        total_seen += n
        n_batches += 1

    assert total_seen == N
    expected_batches = (N + bs - 1) // bs
    assert n_batches == expected_batches
