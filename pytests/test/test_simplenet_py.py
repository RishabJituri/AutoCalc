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


class SimpleNet(ag.nn.Module):
    def __init__(self, in_dim=2, hidden=16, out_dim=1):
        super().__init__()
        self.fc1 = ag.nn.Linear(in_dim, hidden, seed=0)
        self.fc2 = ag.nn.Linear(hidden, out_dim, seed=1)

    def forward(self, x):
        x = self.fc1(x)
        x = ag.relu(x)
        x = self.fc2(x)
        return x


def make_dataset(B, seed=42):
    rng = np.random.RandomState(seed)
    X = rng.uniform(-1.0, 1.0, size=(B, 2)).astype(np.float32)
    # linear target
    y = (2.0 * X[:, 0] - 3.0 * X[:, 1] + 0.5).astype(np.float32)
    return X, y.reshape(B, 1)


def test_simplenet_regression_converges():
    B = 256
    X, Y = make_dataset(B)

    model = SimpleNet()
    x_var = ag.Variable.from_numpy(X)
    y_var = ag.Variable.from_numpy(Y)

    out0 = model.forward(x_var)
    diff0 = out0 - y_var
    L0 = (diff0 * diff0).sum() / float(B)
    loss0 = var_to_scalar(L0)

    opt = ag.nn.SGD(0.05)

    for step in range(300):
        model.zero_grad()
        out = model.forward(x_var)
        diff = out - y_var
        L = (diff * diff).sum() / float(B)
        L.backward()
        opt.step(model)

    out1 = model.forward(x_var)
    diff1 = out1 - y_var
    L1 = (diff1 * diff1).sum() / float(B)
    loss1 = var_to_scalar(L1)

    assert loss1 < loss0 * 0.2, f"Expected >5x MSE reduction: {loss0} -> {loss1}"
