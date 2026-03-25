"""
MiniUNet acceptance test — validates that cat + upsample2d + Sequential +
Conv2d + BatchNorm2d + relu can be composed into a tiny UNet-style
encoder-decoder and that a forward + backward pass works end-to-end.
"""
import numpy as np
import pytest
import ag


class MiniUNet(ag.nn.Module):
    """
    Tiny 2-level UNet:
        encoder1  -> maxpool -> encoder2 -> upsample -> cat(skip) -> decoder1 -> head
    Input : [B, in_ch, H, W]
    Output: [B, n_classes, H, W]
    """

    def __init__(self, in_ch=1, n_classes=3):
        super().__init__()
        # Encoder path
        self.enc1 = ag.nn.Conv2d(in_ch, 8, 3, 3, pad_h=1, pad_w=1)
        self.bn1  = ag.nn.BatchNorm2d(8)
        self.enc2 = ag.nn.Conv2d(8, 16, 3, 3, pad_h=1, pad_w=1)
        self.bn2  = ag.nn.BatchNorm2d(16)

        # Decoder path — after cat we have 16+8=24 channels
        self.dec1 = ag.nn.Conv2d(24, 8, 3, 3, pad_h=1, pad_w=1)
        self.bn3  = ag.nn.BatchNorm2d(8)

        # 1×1 head
        self.head = ag.nn.Conv2d(8, n_classes, 1, 1)

    def forward(self, x):
        # Encoder
        e1 = ag.relu(self.bn1(self.enc1(x)))          # [B,8,H,W]
        p  = ag.nn.MaxPool2d(2, 2)(e1)                 # [B,8,H/2,W/2]
        e2 = ag.relu(self.bn2(self.enc2(p)))           # [B,16,H/2,W/2]

        # Decoder
        up = ag.upsample2d(e2, 2, 2)                   # [B,16,H,W]
        cat = ag.cat([up, e1], axis=1)                  # [B,24,H,W]
        d1 = ag.relu(self.bn3(self.dec1(cat)))         # [B,8,H,W]
        logits = self.head(d1)                          # [B,n_classes,H,W]
        return logits


class TestMiniUNet:
    def test_forward_shape(self):
        """Forward pass produces correct output shape."""
        model = MiniUNet(in_ch=1, n_classes=3)
        x = ag.tensor(np.random.randn(2, 1, 8, 8).astype(np.float32))
        y = model(x)
        assert tuple(y.shape()) == (2, 3, 8, 8)

    def test_backward_grads_exist(self):
        """Backward pass through the whole UNet produces non-zero gradients."""
        model = MiniUNet(in_ch=1, n_classes=2)
        x = ag.Variable.from_numpy(
            np.random.randn(1, 1, 4, 4).astype(np.float32), True
        )
        y = model(x)
        loss = ag.reduce_sum(y)
        loss.backward()

        params = model.parameters()
        assert len(params) > 0, "MiniUNet has no parameters"

        has_grad = False
        for p in params:
            g = np.asarray(p.grad())
            if g.size > 0 and np.any(g != 0):
                has_grad = True
                break
        assert has_grad, "No parameter received a non-zero gradient"

    def test_softmax_output(self):
        """Softmax over channel dim produces valid probabilities."""
        model = MiniUNet(in_ch=1, n_classes=4)
        x = ag.tensor(np.random.randn(1, 1, 4, 4).astype(np.float32))
        logits = model(x)  # [1,4,4,4]
        # Flatten spatial dims, softmax over classes
        flat = ag.reshape(logits, [1, 4, 16])  # [1, n_classes, H*W]
        # Transpose to [1, 16, 4] so last dim = classes for softmax
        # Actually softmax expects last dim, so reshape to [16, 4]
        flat2 = ag.reshape(logits, [16, 4])
        probs = ag.softmax(flat2, axis=-1)
        out = np.asarray(probs.value()).reshape(probs.shape())
        row_sums = out.sum(axis=-1)
        np.testing.assert_allclose(row_sums, np.ones(16), atol=1e-4)
