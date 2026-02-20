"""
test_memory_leak.py — Verify the autograd graph does NOT leak Nodes.

These tests exercise the bugs that previously caused OOM (exit 137):
  C1: shared_ptr cycles in matmul / transpose / at(begin,end)
  C2: no post-backward graph cleanup
  S1: duplicate register_parameter in BatchNorm2d::forward()

We use ag.live_node_count() (atomic counter on Node ctor/dtor) to verify
that after each training step, intermediate Nodes are freed.
"""
import gc
import numpy as np
import pytest
import ag


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _baseline_nodes():
    """Force GC and return current live node count."""
    gc.collect()
    return ag.live_node_count()


def var_to_scalar(v):
    """Extract a Python float from an ag.Variable."""
    raw = v.value() if callable(getattr(v, "value", None)) else v.value
    return float(np.asarray(raw, dtype=np.float32).flat[0])


# ---------------------------------------------------------------------------
# 1. Basic: intermediate nodes freed after backward + del
# ---------------------------------------------------------------------------
class TestBasicNodeLifecycle:

    def test_forward_backward_cleanup(self):
        """After backward + del, only leaf nodes survive."""
        before = _baseline_nodes()

        x = ag.Variable([1.0, 2.0, 3.0], [3], True)
        y = ag.reduce_sum(ag.relu(x))
        y.backward()
        del y
        gc.collect()

        after = ag.live_node_count()
        # Only x's node should be alive above baseline
        assert after - before == 1, f"Expected 1 surviving leaf, got {after - before}"

        del x
        gc.collect()
        assert ag.live_node_count() == before

    def test_no_grad_no_graph(self):
        """Operations under nograd create no graph edges."""
        before = _baseline_nodes()

        x = ag.Variable([1.0, 2.0], [2], False)
        with ag.nograd():
            y = ag.relu(x)
            z = ag.reduce_sum(y)

        # z, y, x are live, but no backward closures => del frees them
        del z, y, x
        gc.collect()
        assert ag.live_node_count() == before


# ---------------------------------------------------------------------------
# 2. Matmul cycle (C1 fix verification)
# ---------------------------------------------------------------------------
class TestMatmulCycle:

    def test_matmul_no_leak(self):
        """matmul backward closure must NOT hold a shared_ptr cycle."""
        before = _baseline_nodes()

        A = ag.Variable(list(np.random.randn(6).astype(np.float32)),
                        [2, 3], True)
        B = ag.Variable(list(np.random.randn(12).astype(np.float32)),
                        [3, 4], True)
        C = ag.matmul(A, B)           # the op that had the cycle
        loss = ag.reduce_sum(C)
        loss.backward()

        del loss, C
        gc.collect()
        # Only the two leaf parameters should survive
        after = ag.live_node_count()
        assert after - before == 2, f"matmul leak: {after - before} nodes alive (expected 2)"

        del A, B
        gc.collect()
        assert ag.live_node_count() == before

    def test_matmul_repeated(self):
        """Repeated matmul forward/backward must not accumulate nodes."""
        before = _baseline_nodes()

        A = ag.Variable(list(np.random.randn(6).astype(np.float32)),
                        [2, 3], True)
        B = ag.Variable(list(np.random.randn(12).astype(np.float32)),
                        [3, 4], True)

        for _ in range(50):
            C = ag.matmul(A, B)
            loss = ag.reduce_sum(C)
            loss.backward()
            del loss, C

        gc.collect()
        after = ag.live_node_count()
        assert after - before == 2, \
            f"matmul repeated leak: {after - before} nodes (expected 2 leaves)"

        del A, B
        gc.collect()
        assert ag.live_node_count() == before


# ---------------------------------------------------------------------------
# 3. Transpose cycle (C1 fix verification)
# ---------------------------------------------------------------------------
class TestTransposeCycle:

    def test_transpose_no_leak(self):
        """transpose backward closure must NOT hold a shared_ptr cycle."""
        before = _baseline_nodes()

        A = ag.Variable(list(np.random.randn(6).astype(np.float32)),
                        [2, 3], True)
        T = ag.transpose(A)
        loss = ag.reduce_sum(T)
        loss.backward()

        del loss, T
        gc.collect()
        after = ag.live_node_count()
        assert after - before == 1, f"transpose leak: {after - before} nodes (expected 1)"

        del A
        gc.collect()
        assert ag.live_node_count() == before


# ---------------------------------------------------------------------------
# 4. Slice/at cycle (C1 fix verification)
# ---------------------------------------------------------------------------
class TestSliceCycle:

    def test_at_begin_end_no_leak(self):
        """at(begin,end) backward closure must NOT hold a shared_ptr cycle."""
        before = _baseline_nodes()

        A = ag.Variable([float(i) for i in range(1, 13)], [3, 4], True)
        # Slice first two rows — at() is a method on Variable
        S = A.at([0, 0], [2, 4])
        loss = ag.reduce_sum(S)
        loss.backward()

        del loss, S
        gc.collect()
        after = ag.live_node_count()
        assert after - before == 1, f"at() leak: {after - before} nodes (expected 1)"

        del A
        gc.collect()
        assert ag.live_node_count() == before


# ---------------------------------------------------------------------------
# 5. Post-backward graph cleanup (C2 fix verification)
# ---------------------------------------------------------------------------
class TestPostBackwardCleanup:

    def test_deep_chain_cleanup(self):
        """A deep chain of ops should be fully freed after backward."""
        before = _baseline_nodes()

        x = ag.Variable([0.5], [1], True)
        y = x
        for _ in range(100):
            y = ag.relu(y)
            y = ag.sigmoid(y)
        loss = ag.reduce_sum(y)
        loss.backward()

        del loss, y
        gc.collect()
        after = ag.live_node_count()
        assert after - before == 1, \
            f"Deep chain leak: {after - before} nodes (expected 1 leaf)"

        del x
        gc.collect()
        assert ag.live_node_count() == before


# ---------------------------------------------------------------------------
# 6. Training loop leak test (the OOM scenario)
# ---------------------------------------------------------------------------
class TestTrainingLoopLeak:

    def test_simplenet_training_bounded_nodes(self):
        """
        Simulate a multi-step training loop (Linear layers + cross_entropy).
        Node count must stay bounded — NOT grow with the number of steps.
        This is the scenario that previously caused OOM on MNIST n=60000.
        """
        B, D_in, D_hid, K = 32, 8, 16, 4
        rng = np.random.RandomState(42)

        # Create model
        fc1 = ag.nn.Linear(D_in, D_hid, seed=0)
        fc2 = ag.nn.Linear(D_hid, K, seed=1)

        class TwoLayerNet(ag.nn.Module):
            def __init__(self):
                super().__init__()
                self.fc1 = fc1
                self.fc2 = fc2

            def forward(self, x):
                x = self.fc1(x)
                x = ag.relu(x)
                x = self.fc2(x)
                return x

        model = TwoLayerNet()
        opt = ag.nn.SGD(0.01)

        # Synthetic data
        X = rng.randn(B, D_in).astype(np.float32)
        Y = rng.randint(0, K, size=B).tolist()
        x_var = ag.Variable.from_numpy(X)

        # Warm-up: 1 step to initialize everything
        model.zero_grad()
        logits = model.forward(x_var)
        loss = ag.nn.cross_entropy(logits, Y)
        loss.backward()
        opt.step(model)
        del logits, loss
        gc.collect()

        baseline = ag.live_node_count()

        # Run 100 training steps
        for step in range(100):
            model.zero_grad()
            logits = model.forward(x_var)
            loss = ag.nn.cross_entropy(logits, Y)
            loss.backward()
            opt.step(model)
            del logits, loss

        gc.collect()
        after = ag.live_node_count()

        # Allow a small margin (a few nodes from Python variable references),
        # but the count must NOT grow proportionally with steps.
        growth = after - baseline
        assert growth <= 5, \
            f"Training loop leaks nodes: {growth} extra after 100 steps (baseline={baseline})"

    def test_matmul_training_loop_bounded(self):
        """
        Training loop using matmul (the op that had shared_ptr cycles).
        Verifies the C1 fix holds under repeated forward/backward.
        """
        rng = np.random.RandomState(123)
        M, K_dim, N = 16, 8, 4

        W = ag.Variable(list(rng.randn(K_dim * N).astype(np.float32)),
                        [K_dim, N], True)

        X_np = rng.randn(M, K_dim).astype(np.float32)
        X = ag.Variable.from_numpy(X_np)

        # Warm-up
        out = ag.matmul(X, W)
        loss = ag.reduce_sum(out)
        loss.backward()
        del out, loss
        gc.collect()
        baseline = ag.live_node_count()

        # 100 iterations of matmul forward/backward
        for _ in range(100):
            out = ag.matmul(X, W)
            loss = ag.reduce_sum(out)
            loss.backward()
            # Simple gradient descent
            w_val = np.array(W.value(), dtype=np.float32)
            w_grad = np.array(W.grad(), dtype=np.float32)
            new_w = (w_val - 0.001 * w_grad).tolist()
            # Update W in-place via its node
            W.zero_grad()
            del out, loss

        gc.collect()
        after = ag.live_node_count()
        growth = after - baseline
        assert growth <= 5, \
            f"Matmul training loop leaks: {growth} extra nodes after 100 steps"


# ---------------------------------------------------------------------------
# 7. ResNet-like architecture leak test (matmul + transpose + conv + bn)
# ---------------------------------------------------------------------------
class TestResNetLikeLeak:

    def test_conv_bn_relu_training_bounded(self):
        """
        Conv2d + BatchNorm2d + ReLU training loop — the architecture
        that triggers OOM on MNIST. Node count must stay bounded.
        """
        B, C_in, H, W = 4, 1, 8, 8
        C_out = 4
        rng = np.random.RandomState(77)

        conv = ag.nn.Conv2d(C_in, C_out, 3, 3, stride_h=1, stride_w=1,
                           pad_h=1, pad_w=1, seed=10)
        bn = ag.nn.BatchNorm2d(C_out)

        class ConvBlock(ag.nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = conv
                self.bn = bn

            def forward(self, x):
                x = self.conv(x)
                x = self.bn(x)
                x = ag.relu(x)
                return x

        model = ConvBlock()
        model.train(True)

        X = rng.randn(B, C_in, H, W).astype(np.float32).flatten().tolist()
        x_var = ag.Variable(X, [B, C_in, H, W])

        # Warm-up
        out = model.forward(x_var)
        loss = ag.reduce_sum(out)
        loss.backward()
        del out, loss
        gc.collect()
        baseline = ag.live_node_count()

        # 50 training steps
        for _ in range(50):
            out = model.forward(x_var)
            loss = ag.reduce_sum(out)
            loss.backward()
            model.zero_grad()
            del out, loss

        gc.collect()
        after = ag.live_node_count()
        growth = after - baseline
        assert growth <= 5, \
            f"Conv-BN-ReLU loop leaks: {growth} extra nodes after 50 steps"


# ---------------------------------------------------------------------------
# 8. RSS growth test (the ultimate OOM canary)
# ---------------------------------------------------------------------------
class TestRSSGrowth:

    def test_rss_bounded_over_training(self):
        """
        Run enough training steps that pre-fix OOM would have started.
        Verify RSS does not grow unboundedly.
        """
        import resource

        B, D_in, D_hid, K = 64, 16, 32, 5
        rng = np.random.RandomState(99)

        fc1 = ag.nn.Linear(D_in, D_hid, seed=0)
        fc2 = ag.nn.Linear(D_hid, K, seed=1)

        class Net(ag.nn.Module):
            def __init__(self):
                super().__init__()
                self.fc1 = fc1
                self.fc2 = fc2

            def forward(self, x):
                x = self.fc1(x)
                x = ag.relu(x)
                x = self.fc2(x)
                return x

        model = Net()
        opt = ag.nn.SGD(0.01)

        X = rng.randn(B, D_in).astype(np.float32)
        Y = rng.randint(0, K, size=B).tolist()
        x_var = ag.Variable.from_numpy(X)

        # Warm-up (10 steps)
        for _ in range(10):
            model.zero_grad()
            logits = model.forward(x_var)
            loss = ag.nn.cross_entropy(logits, Y)
            loss.backward()
            opt.step(model)
            del logits, loss
        gc.collect()

        # macOS: ru_maxrss is in bytes
        rss_before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        nodes_before = ag.live_node_count()

        # Run 500 steps (enough to leak significantly pre-fix)
        for _ in range(500):
            model.zero_grad()
            logits = model.forward(x_var)
            loss = ag.nn.cross_entropy(logits, Y)
            loss.backward()
            opt.step(model)
            del logits, loss

        gc.collect()
        rss_after = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        nodes_after = ag.live_node_count()

        node_growth = nodes_after - nodes_before
        # RSS growth in MB (macOS reports bytes)
        rss_growth_mb = (rss_after - rss_before) / (1024 * 1024)

        # Nodes must stay bounded
        assert node_growth <= 5, \
            f"Node leak over 500 steps: {node_growth} extra nodes"

        # RSS should not grow by more than 50MB over 500 steps
        # (pre-fix, this would grow by hundreds of MB)
        assert rss_growth_mb < 50, \
            f"RSS grew by {rss_growth_mb:.1f} MB over 500 steps"


# ---------------------------------------------------------------------------
# 9. Verify loss actually converges (not just leak-free but correct)
# ---------------------------------------------------------------------------
class TestConvergenceWithFixes:

    def test_simplenet_still_converges(self):
        """
        Ensure the leak fixes don't break gradient correctness.
        A simple 2-layer net on synthetic data must converge.
        """
        B, K = 200, 3
        rng = np.random.RandomState(0)

        X = rng.randn(B, 4).astype(np.float32)
        # Simple separable labels
        Y = (np.argmax(X[:, :K], axis=1)).tolist()

        fc1 = ag.nn.Linear(4, 16, seed=0)
        fc2 = ag.nn.Linear(16, K, seed=1)

        class Net(ag.nn.Module):
            def __init__(self):
                super().__init__()
                self.fc1 = fc1
                self.fc2 = fc2

            def forward(self, x):
                x = self.fc1(x)
                x = ag.relu(x)
                x = self.fc2(x)
                return x

        model = Net()
        opt = ag.nn.SGD(0.05)
        x_var = ag.Variable.from_numpy(X)

        # Initial loss
        logits0 = model.forward(x_var)
        loss0 = var_to_scalar(ag.nn.cross_entropy(logits0, Y))
        del logits0

        # Train
        for _ in range(300):
            model.zero_grad()
            logits = model.forward(x_var)
            loss = ag.nn.cross_entropy(logits, Y)
            loss.backward()
            opt.step(model)
            del logits, loss

        # Final loss
        logits1 = model.forward(x_var)
        loss1 = var_to_scalar(ag.nn.cross_entropy(logits1, Y))
        del logits1

        assert loss1 < loss0 * 0.5, \
            f"Model did not converge: {loss0:.4f} -> {loss1:.4f}"

        # And still no leak
        gc.collect()
        # Just check we're in a reasonable range (model params + input)
        assert ag.live_node_count() < 50, \
            f"Unexpected node count after training: {ag.live_node_count()}"
