#!/usr/bin/env python3
import time
import numpy as np
import ag

# Simple Python Dataset subclass that lazily generates random samples
class RandDataset(ag.data.Dataset):
    def __init__(self, n, x_shape=(1,28,28)):
        super().__init__()
        self._n = n
        self._x_shape = x_shape
    def size(self):
        return self._n
    def get(self, idx):
        # generate random image and a dummy label
        x = np.random.rand(*self._x_shape).astype(np.float32)
        y = np.array([idx % 10], dtype=np.float32)
        ex = ag.data.Example()
        ex.x = ag.Variable.from_numpy(x, requires_grad=False)
        ex.y = ag.Variable.from_numpy(y, requires_grad=False)
        return ex


def stress_test(num_samples=5000, batch_size=64, num_workers=4, prefetch=8):
    ds = RandDataset(num_samples)
    opts = ag.data.DataLoaderOptions()
    opts.batch_size = batch_size
    opts.shuffle = True
    opts.num_workers = num_workers
    opts.prefetch_batches = prefetch

    loader = ag.data.DataLoader(ds, opts)

    print(f"Dataset size={ds.size()}, batch_size={batch_size}, num_workers={num_workers}, prefetch={prefetch}")
    t0 = time.time()
    n_batches = 0
    try:
        for i, batch in enumerate(loader):
            # Do a tiny operation to touch data
            m = ag.reduce_mean(batch.x)
            # materialize to numpy to ensure copies occur
            a = m.value()
            if (i+1) % 50 == 0:
                print(f"iter {i+1}: mean={a[0]:.6f}")
            n_batches += 1
    except Exception as e:
        print("Encountered exception during iteration:", e)
    t1 = time.time()
    print(f"Consumed {n_batches} batches in {t1-t0:.3f}s")

if __name__ == '__main__':
    # tune parameters here if desired
    stress_test(num_samples=5000, batch_size=64, num_workers=4, prefetch=8)
