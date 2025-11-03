import numpy as np
import ag


def _get_numpy_grad(param):
    try:
        g = param.grad() if callable(getattr(param, 'grad', None)) else getattr(param, 'grad', None)
    except Exception:
        g = getattr(param, 'grad', None)
    if g is None:
        return None
    return np.asarray(g, dtype=np.float32)


def _to_numpy_value(param):
    try:
        v = param.value() if callable(getattr(param, 'value', None)) else getattr(param, 'value', None)
        return np.asarray(v, dtype=np.float32)
    except Exception:
        return np.asarray(param, dtype=np.float32)


def test_sgd_plain_update_matches_expected():
    # Simple numeric check: SGD (no momentum, no weight_decay) should perform
    # weight <- weight - lr * grad for each parameter.
    # Build a tiny linear layer (1->1) and a trivial loss so gradient is simple.
    lin = ag.nn.Linear(1, 1, bias=False, init_scale=0.02, seed=12345)

    # Batch of ones -> forward = W * 1, mean over batch yields gradient = 1 * dL/dout
    x = ag.Variable.from_numpy(np.ones((4, 1), dtype=np.float32), True)
    yhat = lin.forward(x)
    loss = ag.reduce_mean(yhat)

    # backward to populate grads
    loss.backward()

    params = list(lin.parameters())
    assert len(params) == 1
    w = params[0]

    g = _get_numpy_grad(w)
    assert g is not None and g.size > 0, "expected non-empty gradient on parameter"

    prev = _to_numpy_value(w).copy()
    lr = 0.1
    opt = ag.nn.SGD(lr)
    opt.step(lin)

    after = _to_numpy_value(w)
    expected = prev - lr * g

    assert np.allclose(after, expected, atol=1e-6, rtol=1e-6), (
        f"SGD numeric update mismatch:\nprev={prev}\ngrad={g}\nexpected={expected}\nafter={after}"
    )
