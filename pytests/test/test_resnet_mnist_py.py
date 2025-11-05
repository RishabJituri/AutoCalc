import os
import numpy as np
import ag
import pytest

# Reuse LocalMNIST from test_data_mnist_local_py if present; otherwise embed minimal IDX reader
from pytests.test.test_data_mnist_local_py import LocalMNIST, var_to_numpy


def var_to_scalar(v):
    try:
        if callable(getattr(v, "value", None)):
            raw = v.value()
        else:
            raw = getattr(v, "value", None)
    except Exception:
        raw = getattr(v, "value", None)

    if raw is not None:
        try:
            arr = np.asarray(raw, dtype=np.float32)
            try:
                return float(arr.item())
            except Exception:
                return float(np.asarray(arr))
        except Exception:
            pass

    try:
        return float(np.asarray(v))
    except Exception:
        return float(0.0)


class SmallResBlock(ag.nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = ag.nn.Conv2d(channels, channels, 3, 3, 1, 1, 1, 1, bias=True, init_scale=0.02, seed=0)
        self.bn1 = ag.nn.BatchNorm2d(channels)
        self.conv2 = ag.nn.Conv2d(channels, channels, 3, 3, 1, 1, 1, 1, bias=True, init_scale=0.02, seed=1)
        self.bn2 = ag.nn.BatchNorm2d(channels)

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = ag.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = out + identity
        out = ag.relu(out)
        return out


class SmallResNet(ag.nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv = ag.nn.Conv2d(1, 8, 3, 3, 1, 1, 1, 1, bias=True, init_scale=0.02, seed=2)
        self.bn = ag.nn.BatchNorm2d(8)
        self.block1 = SmallResBlock(8)
        # Use global average pool over 28x28 to produce 1x1 spatial output
        self.pool = ag.nn.AvgPool2d(28,28,0,0,0,0)
        self.fc = ag.nn.Linear(8, num_classes, seed=3)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = ag.relu(x)
        x = self.block1(x)
        x = self.pool(x)
        # flatten per-sample
        x = ag.flatten(x, 1)
        x = self.fc(x)
        return x


@pytest.mark.slow
def test_resnet_mnist_small_train():
    # local MNIST
    data_root = os.path.abspath(os.path.join(os.getcwd(), "Data"))
    mnist_raw = os.path.join(data_root, "MNIST", "raw", "train-images-idx3-ubyte")
    if not os.path.exists(mnist_raw):
        pytest.skip("Local MNIST not found; skipping")

    N = 1024
    bs = 64
    ds = LocalMNIST(data_root, train=True, max_samples=N)
    opts = ag.data.DataLoaderOptions()
    opts.batch_size = bs
    opts.shuffle = True
    opts.seed = 123
    loader = ag.data.DataLoader(ds, opts)

    model = SmallResNet(num_classes=10)
    opt = ag.nn.SGD(0.05, 0.9)

    # initial eval
    # compute initial CE on entire set (one pass)
    all_logits = []
    all_targets = []
    for batch in loader:
        x = batch.x
        # reshape flattened collate if needed
        try:
            shape = x.shape()
        except Exception:
            shape = None
        if shape is not None and len(shape) == 1:
            x = ag.reshape(x, [int(batch.size), 1, 28, 28])
        logits = model(x)
        all_logits.append(logits)
        y_np = var_to_numpy(batch.y).reshape(-1)
        all_targets.extend(list(map(int, y_np.tolist())))
    # concat logits by stacking in numpy then wrap? simplest: evaluate CE per-batch and average
    loader.reset()

    L0_vals = []
    for batch in loader:
        x = batch.x
        try:
            shape = x.shape()
        except Exception:
            shape = None
        if shape is not None and len(shape) == 1:
            x = ag.reshape(x, [int(batch.size), 1, 28, 28])
        logits = model(x)
        y_np = var_to_numpy(batch.y).reshape(-1)
        targets = list(map(int, y_np.tolist()))
        L = ag.nn.cross_entropy(logits, targets)
        L0_vals.append(var_to_scalar(L))
    loss0 = float(np.mean(L0_vals))

    # train few epochs
    epochs = 3
    for ep in range(epochs):
        for batch in loader:
            x = batch.x
            try:
                shape = x.shape()
            except Exception:
                shape = None
            if shape is not None and len(shape) == 1:
                x = ag.reshape(x, [int(batch.size), 1, 28, 28])
            y_np = var_to_numpy(batch.y).reshape(-1)
            targets = list(map(int, y_np.tolist()))
            model.zero_grad()
            logits = model(x)
            L = ag.nn.cross_entropy(logits, targets)
            L.backward()
            opt.step(model)
        # rewind to change ordering between epochs
        loader.rewind()

    # final eval
    L1_vals = []
    for batch in loader:
        x = batch.x
        try:
            shape = x.shape()
        except Exception:
            shape = None
        if shape is not None and len(shape) == 1:
            x = ag.reshape(x, [int(batch.size), 1, 28, 28])
        logits = model(x)
        y_np = var_to_numpy(batch.y).reshape(-1)
        targets = list(map(int, y_np.tolist()))
        L = ag.nn.cross_entropy(logits, targets)
        L1_vals.append(var_to_scalar(L))
    loss1 = float(np.mean(L1_vals))

    assert loss1 < loss0 * 0.9, f"Expected CE to decrease after training: {loss0} -> {loss1}"
