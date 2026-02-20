#!/usr/bin/env python3
"""
ResNet-on-MNIST apples-to-apples benchmark: AutoCalc (AG) vs PyTorch.

Both frameworks use the **exact same** architecture, weight initialization
(Kaiming-uniform), hyperparameters, data ordering (via shared numpy seed),
and training loop structure so any difference in the loss/accuracy curves
is attributable purely to the framework.

Architecture (per-framework):
  stem:   Conv2d(1→32, 3×3, pad=1) → BN → ReLU
  block1: ResBlock(32)   (two 3×3 convs + BN + skip)
  block2: ResBlock(64)   (projection shortcut 32→64, stride-2 downsample)
  head:   GlobalAvgPool → Linear(64→10)

Metrics collected every `--step-interval` steps:
  • training loss  (cross-entropy)
  • training accuracy  (argmax match)
  • wall-clock time  (cumulative seconds since epoch start)
  • RSS memory  (via resource.getrusage)
  • live node count  (AG only — autograd graph leak detector)

Plots (saved as PNGs):
  1. Loss vs step  (AG & Torch overlaid)
  2. Loss vs wall-clock time
  3. Accuracy vs step
  4. Accuracy vs wall-clock time
  5. RSS memory vs step
  6. Per-epoch summary bar chart

Usage:
  python py_demos/ResNet_MNIST.py --n 60000 --bs 128 --epochs 5 \\
         --lr 0.1 --momentum 0.9 --step-interval 20 --plot --compare-torch
"""
import os, sys, time, argparse, json, math, struct, datetime, gc
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# AG import
# ---------------------------------------------------------------------------
import ag
import tracemalloc, resource

tracemalloc.start()

HAVE_MPL = False
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAVE_MPL = True
except Exception:
    pass

# ===========================  UTILITIES  ===================================

def kaiming_uniform_bound(fan_in: int) -> float:
    """Bound for Uniform(-b, b) matching Kaiming/He init (gain=sqrt(2) for ReLU)."""
    return math.sqrt(6.0 / fan_in)

def conv_kaiming_bound(c_in: int, kh: int, kw: int) -> float:
    return kaiming_uniform_bound(c_in * kh * kw)

def linear_kaiming_bound(in_features: int) -> float:
    return kaiming_uniform_bound(in_features)

def var_to_numpy(v):
    try:
        return np.asarray(v.value())
    except Exception:
        pass
    try:
        return np.asarray(v.numpy())
    except Exception:
        pass
    return np.asarray(v)

def var_to_scalar(v, default=0.0):
    arr = var_to_numpy(v)
    if arr is None:
        return float(default)
    return float(np.asarray(arr).ravel()[0])

def rss_mb():
    """Current RSS in MB (macOS: ru_maxrss is bytes)."""
    ru = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform == "darwin":
        return ru / (1024 * 1024)
    return ru / 1024  # Linux: already in KB

def accuracy_from_logits(logits_np, targets):
    """Compute accuracy from (N, C) logits numpy array and list-of-int targets."""
    if logits_np.ndim == 1:
        nc = len(targets)
        logits_np = logits_np.reshape(nc, -1)
    preds = np.argmax(logits_np, axis=1)
    return int((preds == np.array(targets)).sum()), len(targets)

# ==========================  DATASET  ======================================

class LocalMNIST(ag.data.Dataset):
    """Reads raw IDX files from Data/MNIST/raw (no torchvision dependency)."""
    def __init__(self, root, train=True, max_samples=None):
        super().__init__()
        raw = os.path.join(root, "MNIST", "raw")
        prefix = "train" if train else "t10k"
        img_path = os.path.join(raw, f"{prefix}-images-idx3-ubyte")
        lbl_path = os.path.join(raw, f"{prefix}-labels-idx1-ubyte")
        if not os.path.exists(img_path):
            raise RuntimeError(f"MNIST not found: {img_path}")
        with open(img_path, "rb") as f:
            _, num, rows, cols = struct.unpack(">IIII", f.read(16))
            imgs = np.frombuffer(f.read(), dtype=np.uint8).reshape(num, rows, cols)
        with open(lbl_path, "rb") as f:
            struct.unpack(">II", f.read(8))
            lbls = np.frombuffer(f.read(), dtype=np.uint8)
        if max_samples:
            n = min(int(max_samples), imgs.shape[0])
            imgs, lbls = imgs[:n], lbls[:n]
        self._images = imgs
        self._labels = lbls

    def size(self):
        return int(self._images.shape[0])

    def get(self, idx):
        idx = int(idx)
        arr = self._images[idx].astype(np.float32) / 255.0
        ex = ag.data.Example()
        ex.x = ag.Variable.from_numpy(arr[None, :, :], requires_grad=False)  # [1,28,28]
        ex.y = ag.Variable.from_numpy(
            np.array([int(self._labels[idx])], dtype=np.int64), requires_grad=False
        )
        return ex

# ==========================  AG MODEL  =====================================

class AGResBlock(ag.nn.Module):
    """Pre-activation residual block with optional projection shortcut."""
    def __init__(self, in_ch, out_ch, stride=1, seed=0):
        super().__init__()
        s_conv1 = conv_kaiming_bound(in_ch, 3, 3)
        s_conv2 = conv_kaiming_bound(out_ch, 3, 3)

        self.conv1 = ag.nn.Conv2d(in_ch, out_ch, 3, 3, stride, stride, 1, 1,
                                   bias=False, init_scale=s_conv1, seed=seed)
        self.bn1 = ag.nn.BatchNorm2d(out_ch)
        self.conv2 = ag.nn.Conv2d(out_ch, out_ch, 3, 3, 1, 1, 1, 1,
                                   bias=False, init_scale=s_conv2, seed=seed + 1)
        self.bn2 = ag.nn.BatchNorm2d(out_ch)

        self._use_proj = (in_ch != out_ch) or (stride != 1)
        if self._use_proj:
            s_proj = conv_kaiming_bound(in_ch, 1, 1)
            self.proj = ag.nn.Conv2d(in_ch, out_ch, 1, 1, stride, stride, 0, 0,
                                      bias=False, init_scale=s_proj, seed=seed + 2)
            self.proj_bn = ag.nn.BatchNorm2d(out_ch)

    def forward(self, x):
        identity = x
        out = ag.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self._use_proj:
            identity = self.proj_bn(self.proj(x))
        out = out + identity
        return ag.relu(out)


class AGResNet(ag.nn.Module):
    """
    Small ResNet for 28x28 grayscale:
      stem(1->32) -> block1(32->32) -> block2(32->64, stride=2) -> GAP -> FC(64->10)
    """
    def __init__(self, num_classes=10, seed=42):
        super().__init__()
        s_stem = conv_kaiming_bound(1, 3, 3)
        self.stem_conv = ag.nn.Conv2d(1, 32, 3, 3, 1, 1, 1, 1,
                                       bias=False, init_scale=s_stem, seed=seed)
        self.stem_bn = ag.nn.BatchNorm2d(32)

        self.block1 = AGResBlock(32, 32, stride=1, seed=seed + 10)
        self.block2 = AGResBlock(32, 64, stride=2, seed=seed + 20)

        # After block2 with stride=2: spatial is 14x14
        self.pool = ag.nn.AvgPool2d(14, 14)  # global avg pool -> (N, 64, 1, 1)

        s_fc = linear_kaiming_bound(64)
        self.fc = ag.nn.Linear(64, num_classes, bias=True, init_scale=s_fc, seed=seed + 30)

    def forward(self, x):
        x = ag.relu(self.stem_bn(self.stem_conv(x)))
        x = self.block1(x)
        x = self.block2(x)
        x = self.pool(x)
        x = ag.flatten(x, 1)   # (N, 64)
        return self.fc(x)


# ==========================  AG TRAIN / EVAL  ==============================

def _ensure_4d(x_var, batch_size):
    shp = x_var.shape()
    if len(shp) == 4:
        return x_var
    if len(shp) == 1 and shp[0] == batch_size * 28 * 28:
        return ag.reshape(x_var, [int(batch_size), 1, 28, 28])
    if len(shp) == 2 and shp[1] == 784:
        return ag.reshape(x_var, [int(shp[0]), 1, 28, 28])
    return x_var


def evaluate_ag(model, dataset, batch_size):
    try:
        model.train(False)
    except Exception:
        pass
    opts = ag.data.DataLoaderOptions()
    opts.batch_size = int(batch_size)
    opts.shuffle = False
    loader = ag.data.DataLoader(dataset, opts)

    total_loss, total_correct, total_n = 0.0, 0, 0
    for batch in loader:
        x = _ensure_4d(batch.x, int(batch.size))
        y_np = var_to_numpy(batch.y).reshape(-1)
        targets = list(map(int, y_np.tolist()))
        logits = model(x)
        L = ag.nn.cross_entropy(logits, targets)
        total_loss += var_to_scalar(L) * len(targets)
        c, n = accuracy_from_logits(var_to_numpy(logits), targets)
        total_correct += c
        total_n += n
    avg_loss = total_loss / max(total_n, 1)
    acc = total_correct / max(total_n, 1)
    return avg_loss, acc


StepRecord = dict  # keys: step, loss, acc, time_s, rss_mb, nodes (AG only)


def train_ag(model, dataset, batch_size, epochs, lr, momentum, weight_decay,
             seed, step_interval, log_interval=20):
    opt = ag.nn.SGD(lr, momentum, False, weight_decay)
    opts = ag.data.DataLoaderOptions()
    opts.batch_size = int(batch_size)
    opts.shuffle = True
    opts.seed = int(seed)
    loader = ag.data.DataLoader(dataset, opts)

    records: list[StepRecord] = []
    epoch_summaries = []
    global_step = 0

    try:
        model.train(True)
    except Exception:
        pass

    t0 = time.time()

    for ep in range(epochs):
        ep_loss, ep_correct, ep_n = 0.0, 0, 0
        ep_t0 = time.time()

        for bi, batch in enumerate(loader):
            x = _ensure_4d(batch.x, int(batch.size))
            y_np = var_to_numpy(batch.y).reshape(-1)
            targets = list(map(int, y_np.tolist()))
            bs_actual = len(targets)

            model.zero_grad()
            logits = model(x)
            L = ag.nn.cross_entropy(logits, targets)
            L.backward()
            opt.step(model)

            lval = var_to_scalar(L)
            c, n = accuracy_from_logits(var_to_numpy(logits), targets)
            ep_loss += lval * n
            ep_correct += c
            ep_n += n
            global_step += 1

            if step_interval > 0 and (global_step % step_interval == 0):
                elapsed = time.time() - t0
                try:
                    nodes = ag.live_node_count()
                except Exception:
                    nodes = -1
                records.append({
                    "step": global_step,
                    "loss": lval,
                    "acc": c / max(n, 1),
                    "time_s": elapsed,
                    "rss_mb": rss_mb(),
                    "nodes": nodes,
                })

            if log_interval > 0 and ((bi + 1) % log_interval == 0):
                print(f"  AG  ep {ep+1}/{epochs}  batch {bi+1:>4d}  "
                      f"loss={lval:.4f}  acc={c/max(n,1):.3f}")

            del x, y_np, targets, logits, L

            if global_step % 200 == 0:
                gc.collect()

        loader.rewind()
        ep_avg_loss = ep_loss / max(ep_n, 1)
        ep_acc = ep_correct / max(ep_n, 1)
        ep_time = time.time() - ep_t0
        epoch_summaries.append({
            "epoch": ep + 1, "loss": ep_avg_loss, "acc": ep_acc, "time_s": ep_time,
        })
        print(f"AG  epoch {ep+1}/{epochs}  loss={ep_avg_loss:.4f}  "
              f"acc={ep_acc:.4f}  time={ep_time:.1f}s")

    total_time = time.time() - t0
    return total_time, records, epoch_summaries


# ==========================  TORCH MODEL  ==================================

def build_torch_model_and_train(root, args, shared_indices_per_epoch):
    """
    Build a Torch mirror model with *identical* architecture, Kaiming-uniform
    init (PyTorch default for Conv2d/Linear), and train using the same batch
    order (shared_indices_per_epoch generated from the same numpy seed).
    """
    import torch
    import torch.nn as tnn
    import torch.nn.functional as F
    import torch.optim as toptim
    from torch.utils.data import Dataset as TDataset, DataLoader as TDL, Sampler

    torch.manual_seed(args.seed)

    # --- dataset ---
    class TorchMNIST(TDataset):
        def __init__(self, root, train=True, max_samples=None):
            raw = os.path.join(root, "MNIST", "raw")
            prefix = "train" if train else "t10k"
            img_path = os.path.join(raw, f"{prefix}-images-idx3-ubyte")
            lbl_path = os.path.join(raw, f"{prefix}-labels-idx1-ubyte")
            with open(img_path, "rb") as f:
                _, num, rows, cols = struct.unpack(">IIII", f.read(16))
                imgs = np.frombuffer(f.read(), dtype=np.uint8).reshape(num, rows, cols)
            with open(lbl_path, "rb") as f:
                struct.unpack(">II", f.read(8))
                lbls = np.frombuffer(f.read(), dtype=np.uint8)
            if max_samples:
                n = min(int(max_samples), imgs.shape[0])
                imgs, lbls = imgs[:n], lbls[:n]
            self.x = torch.from_numpy((imgs.astype(np.float32) / 255.0)[:, None, :, :])
            self.y = torch.from_numpy(lbls.astype(np.int64))
        def __len__(self):
            return int(self.x.shape[0])
        def __getitem__(self, i):
            return self.x[int(i)], self.y[int(i)]

    # --- deterministic sampler using same indices as AG ---
    class FixedSampler(Sampler):
        def __init__(self, indices_list):
            self.indices = indices_list
        def __iter__(self):
            return iter(self.indices)
        def __len__(self):
            return len(self.indices)

    # --- model (exact mirror of AGResNet) ---
    class TorchResBlock(tnn.Module):
        def __init__(self, in_ch, out_ch, stride=1):
            super().__init__()
            self.conv1 = tnn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)
            self.bn1 = tnn.BatchNorm2d(out_ch)
            self.conv2 = tnn.Conv2d(out_ch, out_ch, 3, stride=1, padding=1, bias=False)
            self.bn2 = tnn.BatchNorm2d(out_ch)
            self.proj = None
            if in_ch != out_ch or stride != 1:
                self.proj = tnn.Sequential(
                    tnn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False),
                    tnn.BatchNorm2d(out_ch),
                )
        def forward(self, x):
            identity = x
            out = F.relu(self.bn1(self.conv1(x)))
            out = self.bn2(self.conv2(out))
            if self.proj is not None:
                identity = self.proj(x)
            return F.relu(out + identity)

    class TorchResNet(tnn.Module):
        def __init__(self, num_classes=10):
            super().__init__()
            self.stem_conv = tnn.Conv2d(1, 32, 3, stride=1, padding=1, bias=False)
            self.stem_bn = tnn.BatchNorm2d(32)
            self.block1 = TorchResBlock(32, 32, stride=1)
            self.block2 = TorchResBlock(32, 64, stride=2)
            self.pool = tnn.AdaptiveAvgPool2d(1)
            self.fc = tnn.Linear(64, num_classes)
        def forward(self, x):
            x = F.relu(self.stem_bn(self.stem_conv(x)))
            x = self.block1(x)
            x = self.block2(x)
            x = self.pool(x)
            x = torch.flatten(x, 1)
            return self.fc(x)

    # --- helpers ---
    def evaluate_torch(net, ds, bs):
        net.eval()
        loader = TDL(ds, batch_size=int(bs), shuffle=False, num_workers=0)
        criterion = tnn.CrossEntropyLoss()
        total_loss, total_correct, total_n = 0.0, 0, 0
        with torch.no_grad():
            for xb, yb in loader:
                out = net(xb)
                loss = criterion(out, yb)
                bs_act = int(yb.shape[0])
                total_loss += float(loss.item()) * bs_act
                total_correct += int((out.argmax(1) == yb).sum().item())
                total_n += bs_act
        return total_loss / max(total_n, 1), total_correct / max(total_n, 1)

    # --- build & train ---
    ds_t = TorchMNIST(root, train=True, max_samples=args.n)
    net = TorchResNet(num_classes=10)
    # Kaiming init is PyTorch default for Conv2d; Linear uses kaiming_uniform_ too.
    # So no explicit init needed — it matches our AG Kaiming-uniform.

    optimizer = toptim.SGD(net.parameters(), lr=args.lr, momentum=args.momentum,
                           weight_decay=args.weight_decay)
    criterion = tnn.CrossEntropyLoss()

    # Initial eval
    loss_init, acc_init = evaluate_torch(net, ds_t, args.bs)
    print(f"Torch initial  loss={loss_init:.4f}  acc={acc_init:.4f}")

    net.train()
    records: list[StepRecord] = []
    epoch_summaries = []
    global_step = 0
    t0 = time.time()

    for ep in range(args.epochs):
        # Use the *same* batch ordering as AG for this epoch
        indices = shared_indices_per_epoch[ep]
        sampler = FixedSampler(indices)
        loader = TDL(ds_t, batch_size=int(args.bs), sampler=sampler, num_workers=0,
                      drop_last=False)

        ep_loss, ep_correct, ep_n = 0.0, 0, 0
        ep_t0 = time.time()

        for bi, (xb, yb) in enumerate(loader):
            optimizer.zero_grad()
            out = net(xb)
            loss = criterion(out, yb)
            loss.backward()
            optimizer.step()

            lval = float(loss.item())
            bs_act = int(yb.shape[0])
            c = int((out.argmax(1) == yb).sum().item())
            ep_loss += lval * bs_act
            ep_correct += c
            ep_n += bs_act
            global_step += 1

            if args.step_interval > 0 and (global_step % args.step_interval == 0):
                elapsed = time.time() - t0
                records.append({
                    "step": global_step,
                    "loss": lval,
                    "acc": c / max(bs_act, 1),
                    "time_s": elapsed,
                    "rss_mb": rss_mb(),
                    "nodes": -1,
                })

            if args.log_interval > 0 and ((bi + 1) % args.log_interval == 0):
                print(f"  Torch  ep {ep+1}/{args.epochs}  batch {bi+1:>4d}  "
                      f"loss={lval:.4f}  acc={c/max(bs_act,1):.3f}")

        ep_avg_loss = ep_loss / max(ep_n, 1)
        ep_acc = ep_correct / max(ep_n, 1)
        ep_time = time.time() - ep_t0
        epoch_summaries.append({
            "epoch": ep + 1, "loss": ep_avg_loss, "acc": ep_acc, "time_s": ep_time,
        })
        print(f"Torch  epoch {ep+1}/{args.epochs}  loss={ep_avg_loss:.4f}  "
              f"acc={ep_acc:.4f}  time={ep_time:.1f}s")

    total_time = time.time() - t0
    loss_final, acc_final = evaluate_torch(net, ds_t, args.bs)
    print(f"Torch final  loss={loss_final:.4f}  acc={acc_final:.4f}  "
          f"total_time={total_time:.1f}s")

    return {
        "loss_before": loss_init, "acc_before": acc_init,
        "loss_after": loss_final, "acc_after": acc_final,
        "time_s": total_time,
        "records": records,
        "epoch_summaries": epoch_summaries,
    }


# ==========================  PLOTTING  =====================================

def make_plots(summary, out_dir: Path):
    if not HAVE_MPL:
        print("matplotlib not available — skipping plots")
        return
    out_dir = Path(out_dir)

    ag_rec = summary.get("ag", {}).get("records", [])
    th_rec = summary.get("torch", {}).get("records", [])
    ag_ep = summary.get("ag", {}).get("epoch_summaries", [])
    th_ep = summary.get("torch", {}).get("epoch_summaries", [])

    def _xy(recs, xkey, ykey):
        return [r[xkey] for r in recs], [r[ykey] for r in recs]

    # ---- 1. Loss vs step ----
    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=140)
    if ag_rec:
        ax.plot(*_xy(ag_rec, "step", "loss"), lw=1.0, label="AG", alpha=0.85)
    if th_rec:
        ax.plot(*_xy(th_rec, "step", "loss"), lw=1.0, label="Torch", alpha=0.85)
    ax.set_xlabel("Training step"); ax.set_ylabel("Cross-entropy loss")
    ax.set_title("Loss vs Step"); ax.legend(); ax.grid(True, alpha=0.3)
    fig.tight_layout(); fig.savefig(out_dir / f"loss_vs_step.png"); plt.close(fig)

    # ---- 2. Loss vs wall-clock time ----
    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=140)
    if ag_rec:
        ax.plot(*_xy(ag_rec, "time_s", "loss"), lw=1.0, label="AG", alpha=0.85)
    if th_rec:
        ax.plot(*_xy(th_rec, "time_s", "loss"), lw=1.0, label="Torch", alpha=0.85)
    ax.set_xlabel("Wall-clock time (s)"); ax.set_ylabel("Cross-entropy loss")
    ax.set_title("Loss vs Time"); ax.legend(); ax.grid(True, alpha=0.3)
    fig.tight_layout(); fig.savefig(out_dir / f"loss_vs_time.png"); plt.close(fig)

    # ---- 3. Accuracy vs step ----
    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=140)
    if ag_rec:
        ax.plot(*_xy(ag_rec, "step", "acc"), lw=1.0, label="AG", alpha=0.85)
    if th_rec:
        ax.plot(*_xy(th_rec, "step", "acc"), lw=1.0, label="Torch", alpha=0.85)
    ax.set_xlabel("Training step"); ax.set_ylabel("Batch accuracy")
    ax.set_title("Accuracy vs Step"); ax.legend(); ax.grid(True, alpha=0.3)
    fig.tight_layout(); fig.savefig(out_dir / f"acc_vs_step.png"); plt.close(fig)

    # ---- 4. Accuracy vs wall-clock time ----
    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=140)
    if ag_rec:
        ax.plot(*_xy(ag_rec, "time_s", "acc"), lw=1.0, label="AG", alpha=0.85)
    if th_rec:
        ax.plot(*_xy(th_rec, "time_s", "acc"), lw=1.0, label="Torch", alpha=0.85)
    ax.set_xlabel("Wall-clock time (s)"); ax.set_ylabel("Batch accuracy")
    ax.set_title("Accuracy vs Time"); ax.legend(); ax.grid(True, alpha=0.3)
    fig.tight_layout(); fig.savefig(out_dir / f"acc_vs_time.png"); plt.close(fig)

    # ---- 5. RSS memory vs step ----
    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=140)
    if ag_rec:
        ax.plot(*_xy(ag_rec, "step", "rss_mb"), lw=1.0, label="AG", alpha=0.85)
    if th_rec:
        ax.plot(*_xy(th_rec, "step", "rss_mb"), lw=1.0, label="Torch", alpha=0.85)
    ax.set_xlabel("Training step"); ax.set_ylabel("RSS (MB)")
    ax.set_title("Memory (RSS) vs Step"); ax.legend(); ax.grid(True, alpha=0.3)
    fig.tight_layout(); fig.savefig(out_dir / f"rss_vs_step.png"); plt.close(fig)

    # ---- 6. Per-epoch summary bars ----
    if ag_ep or th_ep:
        n_epochs = max(len(ag_ep), len(th_ep))
        x_pos = np.arange(1, n_epochs + 1)
        width = 0.35
        fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), dpi=140)

        # 6a. epoch loss
        ax = axes[0]
        if ag_ep:
            ax.bar(x_pos[:len(ag_ep)] - width/2, [e["loss"] for e in ag_ep], width, label="AG")
        if th_ep:
            ax.bar(x_pos[:len(th_ep)] + width/2, [e["loss"] for e in th_ep], width, label="Torch")
        ax.set_xlabel("Epoch"); ax.set_ylabel("Avg loss"); ax.set_title("Epoch Loss")
        ax.legend(); ax.grid(True, alpha=0.3, axis="y")

        # 6b. epoch accuracy
        ax = axes[1]
        if ag_ep:
            ax.bar(x_pos[:len(ag_ep)] - width/2, [e["acc"] for e in ag_ep], width, label="AG")
        if th_ep:
            ax.bar(x_pos[:len(th_ep)] + width/2, [e["acc"] for e in th_ep], width, label="Torch")
        ax.set_xlabel("Epoch"); ax.set_ylabel("Accuracy"); ax.set_title("Epoch Accuracy")
        ax.legend(); ax.grid(True, alpha=0.3, axis="y")

        # 6c. epoch wall time
        ax = axes[2]
        if ag_ep:
            ax.bar(x_pos[:len(ag_ep)] - width/2, [e["time_s"] for e in ag_ep], width, label="AG")
        if th_ep:
            ax.bar(x_pos[:len(th_ep)] + width/2, [e["time_s"] for e in th_ep], width, label="Torch")
        ax.set_xlabel("Epoch"); ax.set_ylabel("Time (s)"); ax.set_title("Epoch Wall Time")
        ax.legend(); ax.grid(True, alpha=0.3, axis="y")

        fig.tight_layout()
        fig.savefig(out_dir / f"epoch_summary.png")
        plt.close(fig)

    # ---- 7. Live node count vs step (AG only) ----
    if ag_rec and any(r.get("nodes", -1) >= 0 for r in ag_rec):
        fig, ax = plt.subplots(figsize=(8, 4.5), dpi=140)
        xs = [r["step"] for r in ag_rec if r.get("nodes", -1) >= 0]
        ys = [r["nodes"] for r in ag_rec if r.get("nodes", -1) >= 0]
        ax.plot(xs, ys, lw=1.0, color="tab:red", alpha=0.85)
        ax.set_xlabel("Training step"); ax.set_ylabel("Live Node count")
        ax.set_title("AG Live Nodes vs Step (leak detector)"); ax.grid(True, alpha=0.3)
        fig.tight_layout(); fig.savefig(out_dir / f"ag_nodes_vs_step.png"); plt.close(fig)

    # ---- 8. Smoothed loss vs step (rolling average) ----
    def _smooth(ys, w=15):
        if len(ys) < w:
            return ys
        return list(np.convolve(ys, np.ones(w)/w, mode='valid'))

    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=140)
    if ag_rec:
        sx, sy = _xy(ag_rec, "step", "loss")
        sy_s = _smooth(sy)
        ax.plot(sx[:len(sy_s)], sy_s, lw=1.2, label="AG (smoothed)", alpha=0.9)
    if th_rec:
        sx, sy = _xy(th_rec, "step", "loss")
        sy_s = _smooth(sy)
        ax.plot(sx[:len(sy_s)], sy_s, lw=1.2, label="Torch (smoothed)", alpha=0.9)
    ax.set_xlabel("Training step"); ax.set_ylabel("Cross-entropy loss (smoothed)")
    ax.set_title("Smoothed Loss vs Step"); ax.legend(); ax.grid(True, alpha=0.3)
    fig.tight_layout(); fig.savefig(out_dir / f"loss_smoothed_vs_step.png"); plt.close(fig)

    print(f"Plots saved to {out_dir}")


# ==========================  SHARED INDEX GENERATION  ======================

def generate_epoch_indices(n_samples, batch_size, epochs, seed):
    """
    Pre-generate the shuffled sample indices for each epoch so both AG and
    Torch iterate in the *exact* same order.  Returns list-of-lists.
    """
    rng = np.random.RandomState(seed)
    result = []
    for _ in range(epochs):
        indices = np.arange(n_samples)
        rng.shuffle(indices)
        result.append(indices.tolist())
    return result


# ==========================  MAIN  =========================================

def run(args):
    root = os.path.abspath(os.path.join(os.getcwd(), "Data"))
    if not os.path.isdir(root):
        raise RuntimeError(f"Data directory not found: {root}")

    ds = LocalMNIST(root, train=True, max_samples=args.n)
    n_actual = ds.size()
    print(f"Dataset: MNIST  N={n_actual}  bs={args.bs}  epochs={args.epochs}  "
          f"lr={args.lr}  momentum={args.momentum}  wd={args.weight_decay}  seed={args.seed}")

    # Pre-generate shared batch indices for deterministic comparison
    shared_indices = generate_epoch_indices(n_actual, args.bs, args.epochs, args.seed)

    # ---- AG ----
    np.random.seed(args.seed)
    model = AGResNet(num_classes=10, seed=args.seed)

    loss0, acc0 = evaluate_ag(model, ds, args.bs)
    print(f"AG initial   loss={loss0:.4f}  acc={acc0:.4f}")

    ag_time, ag_records, ag_epochs = train_ag(
        model, ds, args.bs, args.epochs, args.lr, args.momentum,
        args.weight_decay, args.seed, args.step_interval, args.log_interval,
    )

    loss1, acc1 = evaluate_ag(model, ds, args.bs)
    print(f"AG final     loss={loss1:.4f}  acc={acc1:.4f}  total_time={ag_time:.1f}s")

    summary = {
        "n": n_actual, "bs": args.bs, "epochs": args.epochs,
        "lr": args.lr, "momentum": args.momentum, "weight_decay": args.weight_decay,
        "seed": args.seed, "step_interval": args.step_interval,
        "ag": {
            "loss_before": loss0, "acc_before": acc0,
            "loss_after": loss1, "acc_after": acc1,
            "time_s": ag_time,
            "records": ag_records,
            "epoch_summaries": ag_epochs,
        },
    }

    # ---- Torch (optional) ----
    if args.compare_torch:
        try:
            torch_result = build_torch_model_and_train(root, args, shared_indices)
            summary["torch"] = torch_result
        except Exception as e:
            import traceback
            print(f"Torch comparison failed: {e}")
            traceback.print_exc()

    # ---- Output directory: results/resnet_demo/<timestamp>/ ----
    ts = datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S")
    project_root = Path(os.path.abspath(os.path.join(os.getcwd())))
    run_dir = project_root / "results" / "resnet_demo" / ts
    run_dir.mkdir(parents=True, exist_ok=True)

    # Write params.txt
    params_lines = [
        f"ResNet MNIST Benchmark — {ts}",
        f"{'='*50}",
        f"n              = {n_actual}",
        f"batch_size     = {args.bs}",
        f"epochs         = {args.epochs}",
        f"lr             = {args.lr}",
        f"momentum       = {args.momentum}",
        f"weight_decay   = {args.weight_decay}",
        f"seed           = {args.seed}",
        f"step_interval  = {args.step_interval}",
        f"log_interval   = {args.log_interval}",
        f"compare_torch  = {args.compare_torch}",
        f"",
        f"Architecture: stem(1->32) -> ResBlock(32->32) -> ResBlock(32->64,s2) -> GAP -> FC(64->10)",
        f"Init: Kaiming-uniform  bound = sqrt(6/fan_in)",
        f"Optimizer: SGD(lr={args.lr}, momentum={args.momentum}, wd={args.weight_decay})",
        f"",
        f"{'='*50}",
        f"AG Results:",
        f"  loss:  {summary['ag']['loss_before']:.4f} -> {summary['ag']['loss_after']:.4f}",
        f"  acc:   {summary['ag']['acc_before']:.4f} -> {summary['ag']['acc_after']:.4f}",
        f"  time:  {summary['ag']['time_s']:.1f}s",
    ]
    if "torch" in summary:
        params_lines += [
            f"",
            f"Torch Results:",
            f"  loss:  {summary['torch']['loss_before']:.4f} -> {summary['torch']['loss_after']:.4f}",
            f"  acc:   {summary['torch']['acc_before']:.4f} -> {summary['torch']['acc_after']:.4f}",
            f"  time:  {summary['torch']['time_s']:.1f}s",
        ]
    (run_dir / "params.txt").write_text("\n".join(params_lines) + "\n")
    print(f"Params written to {run_dir / 'params.txt'}")

    # ---- Plots ----
    if args.plot:
        make_plots(summary, run_dir)

    # ---- Persist full JSON ----
    out_file = run_dir / "results.json"
    with open(out_file, "w") as f:
        json.dump(summary, f, indent=2, default=float)
    print(f"Results written to {out_file}")

    # ---- Final comparison table ----
    print("\n" + "=" * 70)
    print(f"{'':>20s}  {'AG':>12s}", end="")
    if "torch" in summary:
        print(f"  {'Torch':>12s}", end="")
    print()
    print("-" * 70)
    for key, fmt in [("loss_before", ".4f"), ("loss_after", ".4f"),
                     ("acc_before", ".4f"), ("acc_after", ".4f"),
                     ("time_s", ".1f")]:
        ag_val = summary["ag"].get(key, float("nan"))
        line = f"{key:>20s}  {ag_val:>12{fmt}}"
        if "torch" in summary:
            th_val = summary["torch"].get(key, float("nan"))
            line += f"  {th_val:>12{fmt}}"
        print(line)
    print("=" * 70)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="ResNet MNIST: AG vs Torch benchmark")
    p.add_argument("--n", type=int, default=60000, help="Max training samples")
    p.add_argument("--bs", type=int, default=128, help="Batch size")
    p.add_argument("--epochs", type=int, default=5, help="Training epochs")
    p.add_argument("--lr", type=float, default=0.1, help="Learning rate")
    p.add_argument("--momentum", type=float, default=0.9, help="SGD momentum")
    p.add_argument("--weight-decay", type=float, default=1e-4, help="L2 weight decay")
    p.add_argument("--seed", type=int, default=42, help="Random seed")
    p.add_argument("--step-interval", type=int, default=20,
                   help="Record metrics every N steps (0=disable)")
    p.add_argument("--log-interval", type=int, default=50,
                   help="Print to console every N batches (0=disable)")
    p.add_argument("--plot", action="store_true", help="Save PNG plots")
    p.add_argument("--compare-torch", action="store_true",
                   help="Also train identical Torch model for comparison")
    args = p.parse_args()
    run(args)
