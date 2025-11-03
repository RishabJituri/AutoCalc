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


def run_sgd_and_check(mu, wd, use_nesterov):
    # single-parameter linear
    lin = ag.nn.Linear(1, 1, bias=False, init_scale=0.0, seed=123456)
    x = ag.Variable.from_numpy(np.ones((4, 1), dtype=np.float32), True)
    yhat = lin.forward(x)
    loss = ag.reduce_mean(yhat)
    loss.backward()

    params = list(lin.parameters())
    assert len(params) == 1
    w = params[0]

    g = _get_numpy_grad(w)
    assert g is not None and g.size > 0
    prev = _to_numpy_value(w).copy()

    # initial velocity assumed zero
    v_prev = np.zeros_like(prev)

    # compute expected update following sgd.cpp implementation
    # gi = g + wd * prev (if wd != 0)
    if wd != 0.0:
        gi = g + wd * prev
    else:
        gi = g
    vi = mu * v_prev + gi
    if use_nesterov:
        step = mu * vi + gi
    else:
        step = vi
    lr = 0.1
    expected = prev - lr * step

    opt = ag.nn.SGD(lr, mu, use_nesterov, wd)
    opt.step(lin)

    after = _to_numpy_value(w)
    assert np.allclose(after, expected, atol=1e-6, rtol=1e-6), f"SGD(momentum={mu},wd={wd},nesterov={use_nesterov}) failed: prev={prev}, grad={g}, expected={expected}, after={after}"


def test_sgd_momentum_and_nesterov_variants():
    combos = [
        (0.9, 0.0, False),
        (0.9, 0.01, False),
        (0.9, 0.01, True),
        (0.0, 0.0, False),
    ]
    for mu, wd, use_n in combos:
        run_sgd_and_check(mu, wd, use_n)
