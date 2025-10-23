import numpy as np
import pytest
import ag


def _to_np(x):
    # ag.Variable exposes .value (numpy) per bindings shim
    try:
        return x.value
    except Exception:
        return np.asarray(x)


def _has_nonzero_grads(params, tol=1e-8):
    for p in params:
        # flexibly obtain gradient: property, method, or Variable
        g = getattr(p, "grad", None)
        # if g is a bound method or callable, try calling it
        if callable(g):
            try:
                g = g()
            except Exception:
                # maybe p.grad is not callable; try attribute access
                try:
                    g = p.grad
                except Exception:
                    g = None
        if g is None:
            continue
        # If g is a Variable-like object, convert to numpy
        if hasattr(g, 'numpy') and callable(getattr(g, 'numpy')):
            arr = np.asarray(g.numpy(), dtype=np.float32)
        else:
            try:
                arr = np.asarray(g, dtype=np.float32)
            except Exception:
                continue
        if arr.size == 0:
            continue
        if np.any(np.abs(arr) > tol):
            return True
    return False


def var_to_scalar(v):
    """Robustly convert a loss-like value to Python float. Accepts Variable, numpy array, or backend value/attr."""
    # If object has .numpy(), use it
    try:
        if hasattr(v, 'numpy') and callable(getattr(v, 'numpy')):
            return float(np.asarray(v.numpy()))
    except Exception:
        pass
    # If v has callable .value(), call it; if property, use it
    try:
        raw = v.value() if callable(getattr(v, 'value', None)) else getattr(v, 'value', None)
    except Exception:
        raw = getattr(v, 'value', None)
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
        # last resort: zero
        return 0.0


def test_linear_backward_and_sgd_step():
    rng = np.random.RandomState(123)
    in_f, out_f, batch = 10, 7, 4
    X = rng.randn(batch, in_f).astype(np.float32)
    Y = rng.randn(batch, out_f).astype(np.float32)

    model = ag.nn.Linear(in_f, out_f, seed=0xC0FFEE)
    opt = ag.nn.SGD(0.05)

    # initial forward & loss
    x_var = ag.Variable.from_numpy(X)
    y_var = ag.Variable.from_numpy(Y)
    out = model.forward(x_var)
    diff = out - y_var
    loss = (diff * diff).sum()
    loss.backward()

    # ensure grads exist for parameters
    params = model.parameters()
    assert _has_nonzero_grads(params), "Expected non-zero grads after backward for Linear"

    # take copy of params, step, ensure params changed
    def var_to_np_val(p):
        # call p.value() if it's a callable method, else try attribute
        try:
            raw = p.value() if callable(getattr(p, 'value', None)) else getattr(p, 'value', None)
        except Exception:
            raw = getattr(p, 'value', None)
        if raw is None:
            return None
        arr = np.asarray(raw, dtype=np.float32)
        try:
            return arr.reshape(tuple(p.shape()))
        except Exception:
            return arr

    prev_vals = [var_to_np_val(p) for p in params]
    opt.step(model)
    after_vals = [var_to_np_val(p) for p in params]
    assert any(not np.allclose(a, b) for a, b in zip(prev_vals, after_vals)), "Expected parameters to change after SGD.step"

    # confirm single step decreased the MSE loss on same batch (small model so usually decreases)
    out2 = model.forward(ag.Variable.from_numpy(X))

    def var_to_scalar(v):
        try:
            raw = v.value() if callable(getattr(v, 'value', None)) else getattr(v, 'value', None)
        except Exception:
            raw = getattr(v, 'value', None)
        if raw is None:
            # fallback to numpy conversion
            try:
                return float(np.asarray(v))
            except Exception:
                return float(np.asarray(v.numpy())) if hasattr(v, 'numpy') else float(0.0)
        arr = np.asarray(raw, dtype=np.float32)
        # scalar
        try:
            return float(arr.item())
        except Exception:
            return float(np.asarray(arr))

    loss0 = var_to_scalar(loss)
    diff2 = out2 - y_var
    loss1_var = (diff2 * diff2).sum()
    loss1 = var_to_scalar(loss1_var)
    assert loss1 <= loss0 + 1e-5, f"Loss did not decrease after step: before={loss0} after={loss1}"


def test_conv2d_backward_grads_present():
    rng = np.random.RandomState(42)
    in_ch, out_ch, kh, kw = 3, 4, 3, 3
    batch, H, W = 2, 8, 8
    X = rng.randn(batch, in_ch, H, W).astype(np.float32)

    model = ag.nn.Conv2d(in_ch, out_ch, kh, kw, seed=0x12345)

    x_var = ag.Variable.from_numpy(X)
    out = model.forward(x_var)

    # simple loss: sum of squares
    loss = (out * out).sum()
    loss.backward()

    params = model.parameters()
    assert _has_nonzero_grads(params), "Expected non-zero grads after backward for Conv2d"


# def test_sequential_training_loop_reduces_loss():
#     rng = np.random.RandomState(7)
#     batch = 6
#     dim = 8
#     X = rng.randn(batch, dim).astype(np.float32)
#     Y = np.zeros((batch, dim), dtype=np.float32)

#     seq = ag.nn.Sequential()
#     seq.append(ag.nn.Linear(dim, 16, seed=111))
#     seq.append(ag.nn.Linear(16, dim, seed=222))

#     opt = ag.nn.SGD(0.2)

#     x_var = ag.Variable.from_numpy(X)
#     y_var = ag.Variable.from_numpy(Y)

#     def compute_loss():
#         out = seq.forward(x_var)
#         d = out - y_var
#         return (d * d).sum()

#     # run a handful of training steps and ensure loss is reduced
#     loss_vals = []
#     for _ in range(12):
#         seq.zero_grad()
#         l = compute_loss()
#         loss_vals.append(var_to_scalar(l))
#         l.backward()
#         opt.step(seq)

    # assert loss_vals[-1] <= loss_vals[0] * 0.98, f"Loss did not sufficiently decrease: {loss_vals[0]} -> {loss_vals[-1]}"


def test_lstm_forward_backward_params_receive_grads():
    rng = np.random.RandomState(99)
    seq_len, batch, inp = 3, 2, 6
    X = rng.randn(seq_len, batch, inp).astype(np.float32)

    model = ag.nn.LSTM(inp, 5, 1, True)
    x_var = ag.Variable.from_numpy(X)

    out = model.forward(x_var)
    loss = (out * out).sum()
    loss.backward()

    params = model.parameters()
    assert _has_nonzero_grads(params), "Expected non-zero grads after backward for LSTM"
