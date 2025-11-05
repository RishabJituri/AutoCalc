import numpy as np
import ag


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


class SimpleNetCE(ag.nn.Module):
    def __init__(self, in_dim=2, hidden=32, out_dim=3):
        super().__init__()
        self.fc1 = ag.nn.Linear(in_dim, hidden, seed=0)
        self.fc2 = ag.nn.Linear(hidden, out_dim, seed=1)

    def forward(self, x):
        x = self.fc1(x)
        x = ag.relu(x)
        x = self.fc2(x)
        return x


def make_dataset(B, K=3, seed=0):
    rng = np.random.RandomState(seed)
    X = rng.normal(scale=1.0, size=(B, 2)).astype(np.float32)
    # simple linear separable by projection + modulo
    proj = (X[:, 0] * 2.0 + X[:, 1] * -1.0)
    y = (np.floor((proj - proj.min()) / (proj.max() - proj.min() + 1e-8) * K)).astype(int) % K
    return X, y


def test_simplenet_crossentropy_converges():
    B = 300
    K = 3
    X, Y = make_dataset(B, K)

    model = SimpleNetCE(out_dim=K)
    x_var = ag.Variable.from_numpy(X)
    targets = list(map(int, Y.tolist()))

    logits0 = model.forward(x_var)
    L0 = ag.nn.cross_entropy(logits0, targets)
    loss0 = var_to_scalar(L0)

    opt = ag.nn.SGD(0.1)

    for step in range(200):
        model.zero_grad()
        logits = model.forward(x_var)
        L = ag.nn.cross_entropy(logits, targets)
        L.backward()
        opt.step(model)

    logits1 = model.forward(x_var)
    L1 = ag.nn.cross_entropy(logits1, targets)
    loss1 = var_to_scalar(L1)

    assert loss1 < loss0 * 0.6, f"Expected >1.6x CE reduction: {loss0} -> {loss1}"
