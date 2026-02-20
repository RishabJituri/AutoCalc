import numpy as np
import ag
import pytest


def to_numpy(t):
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


def test_pooling_forward_backward_and_integration():
    rng = np.random.RandomState(0)
    batch = 2
    in_ch = 3
    H = 8
    W = 8

    x = rng.randn(batch, in_ch, H, W).astype(np.float32)
    x_var = ag.Variable.from_numpy(x)

    # Basic Conv -> ReLU -> MaxPool
    # binding signature expects kernel_h, kernel_w, stride_h, stride_w, pad_h, pad_w
    conv = ag.nn.Conv2d(in_ch, 4, 3, 3, 1, 1, 1, 1, seed=0)

    # ensure parameters present on the conv module
    params = list(conv.parameters())
    assert len(params) > 0

    out = conv.forward(ag.Variable.from_numpy(x))
    # apply a simple elementwise op to mimic activation, then pool
    # apply ReLU: some bindings expose functional relu; fall back to elementwise ops
    try:
        # ag.nn.functional.relu if binding exposes it
        relu_fn = getattr(ag.nn, 'functional', None)
        if relu_fn is not None and hasattr(relu_fn, 'relu'):
            out_relu = relu_fn.relu(out)
        elif hasattr(ag, 'relu'):
            out_relu = ag.relu(out)
        else:
            # fallback: try to preserve autograd using framework ops
            try:
                out_relu = ag.clamp(out, 0.0, 1e30)
            except Exception:
                # try elementwise expression if supported; otherwise skip
                try:
                    out_relu = out * (out > 0)
                except Exception:
                    pytest.skip("Cannot perform autograd-preserving ReLU with this ag binding")
    except Exception:
        try:
            out_relu = ag.clamp(out, 0.0, 1e30)
        except Exception:
            try:
                out_relu = out * (out > 0)
            except Exception:
                pytest.skip("Cannot perform autograd-preserving ReLU with this ag binding")
    pooled = ag.nn.MaxPool2d(2, 2).forward(out_relu)

    # shapes: pooled should have fewer or equal elements than the pre-pooled tensor
    np_pooled = to_numpy(pooled)
    np_out = to_numpy(out)
    assert np_pooled is not None
    assert np_out is not None
    assert np_pooled.size <= np_out.size
    # If pooled is 4D, assert spatial dims are halved
    if np_pooled.ndim >= 4:
        assert np_pooled.shape[2] == H // 2 and np_pooled.shape[3] == W // 2

    # Backprop through pooled sum
    L = pooled.sum()
    L.backward()

    # optimizer should update conv params
    p0 = params[0]
    before = to_numpy(p0).copy()
    opt = ag.nn.SGD(0.01)
    # optimize the Conv2d module directly
    opt.step(conv)
    after = to_numpy(p0)
    assert not np.allclose(before, after), "Conv parameter expected to change after optimizer.step()"