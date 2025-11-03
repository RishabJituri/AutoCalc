import numpy as np
import ag
import pytest


def make_image_dataset(B, H=8, W=8, seed=42):
    rng = np.random.RandomState(seed)
    X = rng.uniform(0.0, 1.0, size=(B, 1, H, W)).astype(np.float32)
    labels = (X.mean(axis=(1,2,3)) > 0.5).astype(np.int64)
    return X, labels


class BasicBlock(ag.nn.Module):
    def __init__(self, channels, *, init_scale=0.02, seed=0xC0FFEE):
        super().__init__()
        # assign C++ submodules as attributes -> should auto-register via bindings __setattr__
        self.conv1 = ag.nn.Conv2d(channels, channels, 3, 3, pad_h=1, pad_w=1, init_scale=init_scale, seed=seed)
        self.bn1 = ag.nn.BatchNorm2d(channels)
        self.conv2 = ag.nn.Conv2d(channels, channels, 3, 3, pad_h=1, pad_w=1, init_scale=init_scale, seed=seed+1)
        self.bn2 = ag.nn.BatchNorm2d(channels)

    def forward(self, x):
        identity = x
        out = self.conv1.forward(x)
        out = self.bn1.forward(out)
        out = ag.relu(out)
        out = self.conv2.forward(out)
        out = self.bn2.forward(out)
        # If shapes differ user could provide a downsample; here shapes match
        out = out + identity
        return ag.relu(out)


class ResNetPy(ag.nn.Module):
    def __init__(self, in_ch=1, channels=8, nblocks=2, num_classes=2):
        super().__init__()
        self.conv1 = ag.nn.Conv2d(in_ch, channels, 3, 3, pad_h=1, pad_w=1, init_scale=0.1, seed=0xC0FFEE)
        self.bn1 = ag.nn.BatchNorm2d(channels)
        # create and attach blocks as attributes - they should be auto-registered
        for i in range(nblocks):
            setattr(self, f"block{i}", BasicBlock(channels, init_scale=0.1, seed=0xC0FFEE + i))
        self.fc = ag.nn.Linear(channels, num_classes, seed=12345)

    def forward(self, x):
        out = self.conv1.forward(x)
        out = self.bn1.forward(out)
        out = ag.relu(out)
        # call blocks by attribute names to keep parameter registration working
        i = 0
        while True:
            name = f"block{i}"
            if not hasattr(self, name):
                break
            blk = getattr(self, name)
            out = blk.forward(out)
            i += 1
        pooled = ag.reduce_mean(out, axes=[2,3], keepdims=False)
        logits = self.fc.forward(pooled)
        return logits


def test_resnet_module_python_subclass_converges():
    B = 128
    X, y = make_image_dataset(B, H=8, W=8, seed=2023)
    x_var = ag.Variable.from_numpy(X)
    targets = [int(v) for v in y.tolist()]

    net = ResNetPy(in_ch=1, channels=8, nblocks=2, num_classes=2)
    opt = ag.nn.SGD(0.2)

    # Diagnostics: ensure parameters are registered and being updated by the optimizer
    params = net.parameters()
    assert len(params) > 0, f"No parameters registered on Python Module; got {len(params)}"

    # snapshot parameter values
    prev_vals = []
    for p in params:
        try:
            prev_vals.append(np.array(p.value()))
        except Exception:
            prev_vals.append(np.array(p.value))

    # run a single forward/backward and check grads
    net.zero_grad()
    logits_tmp = net.forward(x_var)
    loss_tmp = ag.nn.cross_entropy(logits_tmp, targets)
    loss_tmp.backward()

    has_nonzero_grad = False
    for p in params:
        try:
            g = np.asarray(p.grad()) if callable(getattr(p, 'grad', None)) else np.asarray(p.grad)
        except Exception:
            try:
                g = np.asarray(p.grad)
            except Exception:
                g = np.array([])
        if g.size and np.any(np.abs(g) > 1e-8):
            has_nonzero_grad = True
            break
    assert has_nonzero_grad, "No non-zero gradients found on parameters after backward(); check registration/graph"

    # optimizer step should change parameters
    try:
        opt.step(net)
    except Exception:
        # try stepping per submodule as a fallback
        for name in [k for k in dir(net) if k.startswith('block') or k in ('conv1','bn1','fc')]:
            m = getattr(net, name)
            try:
                opt.step(m)
            except Exception:
                pass

    after_vals = []
    for p in params:
        try:
            after_vals.append(np.array(p.value()))
        except Exception:
            after_vals.append(np.array(p.value))

    params_changed = any((not np.allclose(a, b)) for a, b in zip(prev_vals, after_vals))
    assert params_changed, "Parameters did not change after optimizer.step(); optimizer not updating registered params"

    # initial loss
    with ag.nograd() as _:
        logits0 = net.forward(x_var)
        L0 = ag.nn.cross_entropy(logits0, targets)
        print("here")
        loss0 = float(np.asarray(L0.value()))
        print(loss0)

    if loss0 <= 1e-8:
        pytest.skip("degenerate initial loss; skipping")

    # train
    for step in range(300):
        net.zero_grad()
        logits = net.forward(x_var)
        loss = ag.nn.cross_entropy(logits, targets)
        loss.backward()
        opt.step(net)

    # eval
    with ag.nograd() as _:
        logits = net.forward(x_var)
    z = logits.value()
    correct = 0
    for i in range(B):
        z0 = z[2*i + 0]
        z1 = z[2*i + 1]
        pred = 1 if z1 > z0 else 0
        if pred == int(y[i]):
            correct += 1
    acc = float(correct) / float(B)
    assert acc >= 0.90, f"Expected acc>=0.90, got {acc}"
