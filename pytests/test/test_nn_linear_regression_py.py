import numpy as np
import ag
import pytest


def var_to_scalar(v):
    # Robust conversion for a scalar Variable produced by the backend
    # Prefer calling .value() if it's exposed as a method in the bindings.
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

    # Fallback: try numpy conversion directly
    try:
        return float(np.asarray(v))
    except Exception:
        return float(0.0)


def make_linear_dataset(B, seed=123):
    rng = np.random.RandomState(seed)
    X = rng.uniform(-1.0, 1.0, size=(B, 2)).astype(np.float32)
    # y = 1.5*x0 - 2.0*x1 + 0.3
    y = (1.5 * X[:, 0] - 2.0 * X[:, 1] + 0.3).astype(np.float32)
    return X, y.reshape(B, 1)


def test_linear_regression_converges():
    B = 128
    X, Y = make_linear_dataset(B, seed=1337)

    # Make test deterministic and avoid exact-fit degeneracy by adding tiny label noise
    noise_rng = np.random.RandomState(9999)
    Y_noisy = (Y + (noise_rng.normal(scale=1e-3, size=Y.shape)).astype(np.float32))

    # Fixed model seed for reproducibility
    model = ag.nn.Linear(2, 1, seed=0xC0FFEE)

    x_var = ag.Variable.from_numpy(X)
    y_var = ag.Variable.from_numpy(Y_noisy)

    # initial MSE (mean over batch)
    out0 = model.forward(x_var)
    diff0 = out0 - y_var
    L0 = (diff0 * diff0).sum() / float(B)
    loss0 = var_to_scalar(L0)

    opt = ag.nn.SGD(0.1)

    # train for a number of steps
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

    assert loss1 < loss0 * 0.1, f"Expected >10x MSE reduction: {loss0} -> {loss1}"
