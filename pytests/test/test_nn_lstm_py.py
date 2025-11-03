import numpy as np
import ag
import pytest


def to_numpy(t):
    # Robust conversion helper for binding objects (Variables, ndarrays, scalars)
    try:
        if callable(getattr(t, "value", None)):
            return np.asarray(t.value())
        v = getattr(t, "value", None)
        if v is not None:
            return np.asarray(v)
    except Exception:
        pass
    try:
        return np.asarray(t)
    except Exception:
        return None


def test_lstm_forward_backward_and_param_update():
    rng = np.random.RandomState(42)
    seq_len = 5
    batch = 2
    in_size = 3
    hid = 4

    # Small deterministic input (seq_len, batch, input_size) - bindings accept this ordering
    x = rng.randn(seq_len, batch, in_size).astype(np.float32)
    x_var = ag.Variable.from_numpy(x)

    # Construct LSTM and ensure parameters are exposed
    # binding signature: LSTM(input_size, hidden_size, num_layers=1, bias=True)
    model = ag.nn.LSTM(in_size, hid, 1)
    params = list(model.parameters())
    assert len(params) > 0, "LSTM should expose at least one parameter"

    # Forward: bindings may return (output, (h_n, c_n)) or just output
    out = model.forward(x_var)
    if isinstance(out, tuple) or isinstance(out, list):
        out0 = out[0]
    else:
        out0 = out

    # Reduce to scalar loss and backprop
    L = out0.sum()
    L.backward()

    # Pick one parameter and verify optimizer.step modifies it
    p = params[0]
    before = to_numpy(p)
    assert before is not None, "failed to convert parameter to numpy"
    before = before.copy()

    opt = ag.nn.SGD(0.01)
    opt.step(model)

    after = to_numpy(p)
    assert after is not None, "failed to convert parameter to numpy after step"

    # Expect some change (not necessarily large); tolerate small floating noise but not exact equality
    assert not np.allclose(before, after), "Expected parameter to change after optimizer.step()"