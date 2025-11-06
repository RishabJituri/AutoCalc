#!/usr/bin/env python3
"""
ResNet-on-MNIST benchmark using AG with our own Dataset + DataLoader (no torchvision).
- Reads local IDX files under ./Data/MNIST/raw
- SmallResNet topology matches pytests (Conv-BN-ReLU -> ResBlock -> GAP -> FC)
- Tracks per-epoch and per-batch training losses
- Optional: record per-step losses at a configurable step_interval across epochs
- Optional plotting (headless-safe), with an option to also plot derivatives/improvements
- Optional: Torch comparison with identical topology and combined plots
Usage:
  python benchmarks/ResNet_MNIST.py --n 1024 --bs 64 --epochs 3 --seed 0 --plot --step-interval 10 --plot-deriv --compare-torch
"""
import os
import time
import argparse
import json
import datetime
import struct
from pathlib import Path

import numpy as np
import ag

# Make plotting work in headless runs
HAVE_MATPLOTLIB = False
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAVE_MATPLOTLIB = True
except Exception:
    HAVE_MATPLOTLIB = False


# ---------- Utilities ----------
def var_to_numpy(v):
    try:
        if callable(getattr(v, "value", None)):
            return np.asarray(v.value())
        v_raw = getattr(v, "value", None)
        if v_raw is not None:
            return np.asarray(v_raw)
    except Exception:
        pass
    try:
        if callable(getattr(v, "numpy", None)):
            return np.asarray(v.numpy())
    except Exception:
        pass
    try:
        return np.asarray(v)
    except Exception:
        return None


def var_to_scalar(v, default=0.0):
    arr = var_to_numpy(v)
    if arr is None:
        return float(default)
    try:
        return float(np.asarray(arr).item())
    except Exception:
        try:
            return float(np.asarray(arr).mean())
        except Exception:
            return float(default)


# ---------- Dataset (our own) ----------
class LocalMNIST(ag.data.Dataset):
    """Read local MNIST idx files under Data/MNIST/raw without torchvision."""
    def __init__(self, root, train=True, max_samples=None):
        super().__init__()
        mnist_raw_dir = os.path.join(root, "MNIST", "raw")
        if train:
            img_path = os.path.join(mnist_raw_dir, "train-images-idx3-ubyte")
            lbl_path = os.path.join(mnist_raw_dir, "train-labels-idx1-ubyte")
        else:
            img_path = os.path.join(mnist_raw_dir, "t10k-images-idx3-ubyte")
            lbl_path = os.path.join(mnist_raw_dir, "t10k-labels-idx1-ubyte")
        if not (os.path.exists(img_path) and os.path.exists(lbl_path)):
            raise RuntimeError(f"MNIST idx files not found under {mnist_raw_dir}")
        with open(img_path, 'rb') as f:
            magic, num, rows, cols = struct.unpack('>IIII', f.read(16))
            data = np.frombuffer(f.read(), dtype=np.uint8)
            data = data.reshape(num, rows, cols)
        with open(lbl_path, 'rb') as f:
            magic_l, num_l = struct.unpack('>II', f.read(8))
            labels = np.frombuffer(f.read(), dtype=np.uint8)
        if max_samples:
            num_take = min(max_samples, data.shape[0])
            self._images = data[:num_take]
            self._labels = labels[:num_take]
        else:
            self._images = data
            self._labels = labels

    def size(self):
        return int(self._images.shape[0])

    def get(self, idx):
        idx = int(idx)
        arr = self._images[idx].astype(np.float32) / 255.0  # [28,28]
        if arr.ndim == 2:
            arr = arr[None, :, :]  # [1,28,28]
        ex = ag.data.Example()
        ex.x = ag.Variable.from_numpy(arr, requires_grad=False)
        ex.y = ag.Variable.from_numpy(np.array([int(self._labels[idx])], dtype=np.int64), requires_grad=False)
        return ex


# ---------- Model ----------
class SmallResBlock(ag.nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = ag.nn.Conv2d(channels, channels, 3, 3, 1, 1, 1, 1, bias=True, init_scale=0.02, seed=0)
        self.bn1 = ag.nn.BatchNorm2d(channels)
        self.conv2 = ag.nn.Conv2d(channels, channels, 3, 3, 1, 1, 1, 1, bias=True, init_scale=0.02, seed=1)
        self.bn2 = ag.nn.BatchNorm2d(channels)

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = ag.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = out + identity
        out = ag.relu(out)
        return out


class SmallResNet(ag.nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv = ag.nn.Conv2d(1, 8, 3, 3, 1, 1, 1, 1, bias=True, init_scale=0.02, seed=2)
        self.bn = ag.nn.BatchNorm2d(8)
        self.block1 = SmallResBlock(8)
        # Global average pool over 28x28 -> 1x1
        self.pool = ag.nn.AvgPool2d(28, 28, 0, 0, 0, 0)
        self.fc = ag.nn.Linear(8, num_classes, seed=3)

        # Optional: expose for tests/inspection
        self.num_classes = num_classes

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = ag.relu(x)
        x = self.block1(x)
        x = self.pool(x)
        x = ag.flatten(x, 1)  # [N,8]
        x = self.fc(x)
        return x


# ---------- Training/Eval ----------

def _ensure_4d_batch_x(x_var, batch_size):
    """Handle collate variants that flatten; reshape back to [N,1,28,28] when needed."""
    try:
        shp = x_var.shape()
    except Exception:
        shp = None
    if shp is None:
        return x_var
    # If 1D or [N, 784], reshape
    try:
        if len(shp) == 1 and shp[0] == batch_size * 28 * 28:
            return ag.reshape(x_var, [int(batch_size), 1, 28, 28])
        if len(shp) == 2 and shp[1] == 28 * 28:
            return ag.reshape(x_var, [int(shp[0]), 1, 28, 28])
    except Exception:
        pass
    return x_var


def evaluate_ag_model(model, dataset, batch_size):
    try:
        model.train(False)
    except Exception:
        pass
    opts = ag.data.DataLoaderOptions()
    opts.batch_size = int(batch_size)
    opts.shuffle = False
    loader = ag.data.DataLoader(dataset, opts)

    losses = []
    total = 0
    correct = 0
    for batch in loader:
        x = _ensure_4d_batch_x(batch.x, int(batch.size))
        logits = model(x)
        y_np = var_to_numpy(batch.y).reshape(-1)
        targets = list(map(int, y_np.tolist()))
        L = ag.nn.cross_entropy(logits, targets)
        losses.append(var_to_scalar(L))
        # accuracy
        logits_np = var_to_numpy(logits)
        if logits_np.ndim == 1 and logits_np.size % int(batch.size) == 0:
            logits_np = logits_np.reshape(int(batch.size), logits_np.size // int(batch.size))
        pred = np.argmax(logits_np, axis=1)
        correct += int((pred == targets).sum())
        total += int(batch.size)
    return float(np.mean(losses)), (correct / float(total)) if total else 0.0


def train_ag(model, dataset, batch_size, epochs, lr, momentum, seed=0, log_interval=10, step_interval=0):
    opt = ag.nn.SGD(lr, momentum)
    opts = ag.data.DataLoaderOptions()
    opts.batch_size = int(batch_size)
    opts.shuffle = True
    opts.seed = int(seed)
    loader = ag.data.DataLoader(dataset, opts)

    per_epoch_losses = []
    per_epoch_batch_losses = []
    per_epoch_step_losses = []  # sampled every step_interval
    step_losses_flat = []       # flattened across epochs

    start = time.time()
    try:
        model.train(True)
    except Exception:
        pass

    global_step = 0
    for ep in range(epochs):
        batch_losses = []
        epoch_step_samples = []
        total_batches = 0
        for bi, batch in enumerate(loader):
            x = _ensure_4d_batch_x(batch.x, int(batch.size))
            y_np = var_to_numpy(batch.y).reshape(-1)
            targets = list(map(int, y_np.tolist()))

            model.zero_grad()
            logits = model(x)
            L = ag.nn.cross_entropy(logits, targets)
            L.backward()
            opt.step(model)

            lval = var_to_scalar(L)
            batch_losses.append(lval)
            total_batches += 1

            global_step += 1
            if step_interval and step_interval > 0 and (global_step % int(step_interval) == 0):
                epoch_step_samples.append(lval)
                step_losses_flat.append(lval)

            if log_interval and ((bi + 1) % log_interval == 0):
                print(f"ep {ep+1}/{epochs} batch {bi+1} loss={lval:.4f}")

            # release references quickly
            del x, y_np, targets, logits, L
        # rewind changes order across epochs
        loader.rewind()
        loss_ep = float(np.mean(batch_losses)) if batch_losses else float('nan')
        per_epoch_losses.append(loss_ep)
        per_epoch_batch_losses.append(batch_losses)
        per_epoch_step_losses.append(epoch_step_samples)
        print(f"AG epoch {ep+1}/{epochs}: loss={loss_ep:.4f} batches={total_batches}")

    duration = time.time() - start
    return duration, per_epoch_losses, per_epoch_batch_losses, per_epoch_step_losses, step_losses_flat


# ---------- Plotting ----------
def plot_training_curves(summary, epochs, out_dir: Path, plot_deriv: bool = False):
    ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    out_dir = Path(out_dir)

    if not HAVE_MATPLOTLIB:
        return None

    # Epoch loss plot (AG vs Torch)
    plt.figure(figsize=(6,4), dpi=140)
    xs = list(range(1, int(epochs) + 1))
    ag_losses = summary.get('ag', {}).get('losses', [])
    if ag_losses:
        plt.plot(xs[:len(ag_losses)], ag_losses, marker='o', label='AG train (epoch mean)')
    torch_losses = summary.get('torch', {}).get('losses', [])
    if torch_losses:
        plt.plot(xs[:len(torch_losses)], torch_losses, marker='s', label='Torch train (epoch mean)')
    plt.xlabel('Epoch')
    plt.ylabel('CrossEntropy Loss')
    plt.title('Epoch training loss')
    plt.grid(True, alpha=0.3)
    plt.legend()
    out1 = out_dir / f"results_resnet_{ts}.png"
    plt.tight_layout()
    plt.savefig(out1)

    # Per-batch training plot (flatten all epochs back-to-back) combined
    steps_ag, vals_ag = [], []
    bl_ag = summary.get('ag', {}).get('batch_losses', [])
    st = 0
    for ep_list in bl_ag:
        for v in ep_list:
            st += 1
            steps_ag.append(st)
            vals_ag.append(float(v))
    steps_t, vals_t = [], []
    bl_t = summary.get('torch', {}).get('batch_losses', [])
    stt = 0
    for ep_list in bl_t:
        for v in ep_list:
            stt += 1
            steps_t.append(stt)
            vals_t.append(float(v))
    if steps_ag or steps_t:
        plt.figure(figsize=(6,4), dpi=140)
        if steps_ag:
            plt.plot(steps_ag, vals_ag, lw=0.9, label='AG train (per-batch)')
        if steps_t:
            plt.plot(steps_t, vals_t, lw=0.9, label='Torch train (per-batch)')
        plt.xlabel('Train step (batches)')
        plt.ylabel('CrossEntropy Loss')
        plt.title('Per-batch training loss (combined)')
        plt.grid(True, alpha=0.3)
        plt.legend()
        out2 = out_dir / f"results_resnet_train_steps_{ts}.png"
        plt.tight_layout()
        plt.savefig(out2)

    # Step-sampled training plot (every k steps across epochs) combined
    k = int(summary.get('step_interval', 0) or 0)
    sl_flat_ag = summary.get('ag', {}).get('step_losses_flat', [])
    sl_flat_t = summary.get('torch', {}).get('step_losses_flat', [])
    if sl_flat_ag or sl_flat_t:
        plt.figure(figsize=(6,4), dpi=140)
        if sl_flat_ag:
            xs_ag = list(range(1, len(sl_flat_ag) + 1))
            plt.plot(xs_ag, sl_flat_ag, lw=0.9, marker='.', label=f'AG every {k} steps' if k else 'AG samples')
        if sl_flat_t:
            xs_t = list(range(1, len(sl_flat_t) + 1))
            plt.plot(xs_t, sl_flat_t, lw=0.9, marker='.', label=f'Torch every {k} steps' if k else 'Torch samples')
        plt.xlabel('Sample index')
        plt.ylabel('CrossEntropy Loss')
        title = 'Step-sampled training loss (combined)'
        if k:
            title += f' (interval={k})'
        plt.title(title)
        plt.grid(True, alpha=0.3)
        plt.legend()
        out3 = out_dir / f"results_resnet_train_steps_sampled_{ts}.png"
        plt.tight_layout()
        plt.savefig(out3)

    # Optional: derivative/improvement plots (combined)
    if plot_deriv:
        # Per-batch improvement
        if len(vals_ag) >= 2 or len(vals_t) >= 2:
            plt.figure(figsize=(6,4), dpi=140)
            if len(vals_ag) >= 2:
                imp_ag = [vals_ag[i-1] - vals_ag[i] for i in range(1, len(vals_ag))]
                xs_i_ag = list(range(1, len(imp_ag) + 1))
                plt.plot(xs_i_ag, imp_ag, lw=0.8, label='AG improvement (per-batch)')
            if len(vals_t) >= 2:
                imp_t = [vals_t[i-1] - vals_t[i] for i in range(1, len(vals_t))]
                xs_i_t = list(range(1, len(imp_t) + 1))
                plt.plot(xs_i_t, imp_t, lw=0.8, label='Torch improvement (per-batch)')
            plt.axhline(0.0, color='k', lw=0.8, alpha=0.5)
            plt.xlabel('Train step (batches)')
            plt.ylabel('Loss improvement (L[t-1]-L[t])')
            plt.title('Per-batch improvement (combined)')
            plt.grid(True, alpha=0.3)
            plt.legend()
            out_imp = out_dir / f"results_resnet_train_steps_improvement_{ts}.png"
            plt.tight_layout()
            plt.savefig(out_imp)
        # Step-sampled improvement
        if len(sl_flat_ag) >= 2 or len(sl_flat_t) >= 2:
            plt.figure(figsize=(6,4), dpi=140)
            if len(sl_flat_ag) >= 2:
                simp_ag = [sl_flat_ag[i-1] - sl_flat_ag[i] for i in range(1, len(sl_flat_ag))]
                xs_s_ag = list(range(1, len(simp_ag) + 1))
                plt.plot(xs_s_ag, simp_ag, lw=0.9, marker='.', label='AG improvement (sampled)')
            if len(sl_flat_t) >= 2:
                simp_t = [sl_flat_t[i-1] - sl_flat_t[i] for i in range(1, len(sl_flat_t))]
                xs_s_t = list(range(1, len(simp_t) + 1))
                plt.plot(xs_s_t, simp_t, lw=0.9, marker='.', label='Torch improvement (sampled)')
            plt.axhline(0.0, color='k', lw=0.8, alpha=0.5)
            plt.xlabel('Sample index')
            plt.ylabel('Loss improvement (L[t-1]-L[t])')
            title = 'Step-sampled improvement (combined)'
            if k:
                title += f' (interval={k})'
            plt.title(title)
            plt.grid(True, alpha=0.3)
            plt.legend()
            out_simp = out_dir / f"results_resnet_train_steps_sampled_improvement_{ts}.png"
            plt.tight_layout()
            plt.savefig(out_simp)

    return out1


# ---------- Runner ----------
def run(args):
    root = os.path.abspath(os.path.join(os.getcwd(), "Data"))
    if not os.path.isdir(root):
        raise RuntimeError(f"Data directory not found: {root}")

    ds = LocalMNIST(root, train=True, max_samples=args.n)
    print(f"Loaded MNIST Dataset: N={ds.size()} bs={args.bs} epochs={args.epochs}")

    np.random.seed(args.seed)
    model = SmallResNet(num_classes=10)

    # Initial eval
    loss0, acc0 = evaluate_ag_model(model, ds, args.bs)

    # Train
    t_train, epoch_losses, batch_losses, step_losses, step_losses_flat = train_ag(
        model, ds, args.bs, args.epochs, args.lr, args.momentum, seed=args.seed, step_interval=args.step_interval
    )

    # Final eval
    loss1, acc1 = evaluate_ag_model(model, ds, args.bs)

    print(f"AG model: loss {loss0:.4f}->{loss1:.4f}, acc {acc0:.3f}->{acc1:.3f}, train_time {t_train:.2f}s")

    summary = {
        'n': int(args.n), 'bs': int(args.bs), 'epochs': int(args.epochs), 'seed': int(args.seed),
        'step_interval': int(args.step_interval),
        'ag': {
            'loss_before': float(loss0), 'loss_after': float(loss1),
            'acc_before': float(acc0), 'acc_after': float(acc1),
            'time_s': float(t_train), 'losses': list(map(float, epoch_losses)),
            'batch_losses': [[float(x) for x in bl] for bl in batch_losses],
            'step_losses': [[float(x) for x in sl] for sl in step_losses],
            'step_losses_flat': [float(x) for x in step_losses_flat],
        },
    }

    # Optional Torch comparison
    if getattr(args, 'compare_torch', False):
        try:
            import torch
            import torch.nn as nn
            import torch.optim as optim
            from torch.utils.data import Dataset, DataLoader
            torch.manual_seed(args.seed)
            init_scale = 0.02

            # Torch-native MNIST dataset reading IDX files directly
            class TorchLocalMNIST(Dataset):
                def __init__(self, root, train=True, max_samples=None):
                    mnist_raw_dir = os.path.join(root, "MNIST", "raw")
                    if train:
                        img_path = os.path.join(mnist_raw_dir, "train-images-idx3-ubyte")
                        lbl_path = os.path.join(mnist_raw_dir, "train-labels-idx1-ubyte")
                    else:
                        img_path = os.path.join(mnist_raw_dir, "t10k-images-idx3-ubyte")
                        lbl_path = os.path.join(mnist_raw_dir, "t10k-labels-idx1-ubyte")
                    if not (os.path.exists(img_path) and os.path.exists(lbl_path)):
                        raise RuntimeError(f"MNIST idx files not found under {mnist_raw_dir}")
                    with open(img_path, 'rb') as f:
                        _magic, num, rows, cols = struct.unpack('>IIII', f.read(16))
                        data = np.frombuffer(f.read(), dtype=np.uint8)
                        data = data.reshape(num, rows, cols)
                    with open(lbl_path, 'rb') as f:
                        _magic_l, num_l = struct.unpack('>II', f.read(8))
                        labels = np.frombuffer(f.read(), dtype=np.uint8)
                    if max_samples:
                        take = min(int(max_samples), data.shape[0])
                        data = data[:take]
                        labels = labels[:take]
                    data = (data.astype(np.float32) / 255.0)[:, None, :, :]
                    labels = labels.astype(np.int64)
                    self.x = torch.from_numpy(data)
                    self.y = torch.from_numpy(labels)
                def __len__(self):
                    return int(self.x.shape[0])
                def __getitem__(self, idx):
                    i = int(idx)
                    return self.x[i], self.y[i]

            def evaluate_torch_model(net, dataset, batch_size):
                net.eval()
                loader = DataLoader(dataset, batch_size=int(batch_size), shuffle=False, num_workers=0)
                criterion = nn.CrossEntropyLoss()
                losses = []
                correct = 0
                n = 0
                with torch.no_grad():
                    for xb, yb in loader:
                        out = net(xb)
                        loss = criterion(out, yb)
                        losses.append(float(loss.item()))
                        pred = out.argmax(dim=1)
                        correct += int((pred == yb).sum().item())
                        n += int(yb.shape[0])
                return (float(np.mean(losses)) if losses else float('nan'), (correct / float(n)) if n else 0.0)

            class TorchSmallResBlock(nn.Module):
                def __init__(self, ch):
                    super().__init__()
                    self.conv1 = nn.Conv2d(ch, ch, kernel_size=3, stride=1, padding=1, bias=True)
                    self.bn1 = nn.BatchNorm2d(ch)
                    self.conv2 = nn.Conv2d(ch, ch, kernel_size=3, stride=1, padding=1, bias=True)
                    self.bn2 = nn.BatchNorm2d(ch)
                    # init
                    for m in [self.conv1, self.conv2]:
                        nn.init.normal_(m.weight, mean=0.0, std=init_scale)
                        if m.bias is not None:
                            nn.init.zeros_(m.bias)
                    nn.init.ones_(self.bn1.weight); nn.init.zeros_(self.bn1.bias)
                    nn.init.ones_(self.bn2.weight); nn.init.zeros_(self.bn2.bias)
                def forward(self, x):
                    identity = x
                    out = self.conv1(x)
                    out = self.bn1(out)
                    out = nn.functional.relu(out)
                    out = self.conv2(out)
                    out = self.bn2(out)
                    out = out + identity
                    out = nn.functional.relu(out)
                    return out

            class TorchSmallResNet(nn.Module):
                def __init__(self, num_classes=10):
                    super().__init__()
                    self.conv = nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=1, bias=True)
                    self.bn = nn.BatchNorm2d(8)
                    self.block1 = TorchSmallResBlock(8)
                    self.pool = nn.AvgPool2d(28)
                    self.fc = nn.Linear(8, num_classes)
                    # init
                    nn.init.normal_(self.conv.weight, mean=0.0, std=init_scale)
                    if self.conv.bias is not None:
                        nn.init.zeros_(self.conv.bias)
                    nn.init.ones_(self.bn.weight); nn.init.zeros_(self.bn.bias)
                    nn.init.normal_(self.fc.weight, mean=0.0, std=init_scale)
                    if self.fc.bias is not None:
                        nn.init.zeros_(self.fc.bias)
                def forward(self, x):
                    x = self.conv(x)
                    x = self.bn(x)
                    x = nn.functional.relu(x)
                    x = self.block1(x)
                    x = self.pool(x)
                    x = torch.flatten(x, 1)
                    x = self.fc(x)
                    return x

            # Build Torch dataset/loader
            ds_t = TorchLocalMNIST(root, train=True, max_samples=args.n)
            g = torch.Generator()
            g.manual_seed(args.seed)
            loader_t = DataLoader(ds_t, batch_size=int(args.bs), shuffle=True, num_workers=0, generator=g)

            net = TorchSmallResNet(num_classes=10)
            opt_t = optim.SGD(net.parameters(), lr=args.lr, momentum=args.momentum)
            criterion = nn.CrossEntropyLoss()

            # Initial eval (full dataset)
            loss_init, acc_init = evaluate_torch_model(net, ds_t, args.bs)

            net.train()
            t0 = time.time()
            per_epoch_losses_t = []
            per_epoch_batch_losses_t = []
            per_epoch_step_losses_t = []
            step_losses_flat_t = []
            global_step_t = 0
            log_interval_t = 10
            for ep in range(args.epochs):
                batch_losses_t = []
                step_samples_ep = []
                total_batches_t = 0
                for bi, (xb, yb) in enumerate(loader_t):
                    opt_t.zero_grad()
                    out = net(xb)
                    loss = criterion(out, yb)
                    loss.backward()
                    opt_t.step()

                    lval = float(loss.item())
                    batch_losses_t.append(lval)
                    total_batches_t += 1

                    global_step_t += 1
                    if args.step_interval and args.step_interval > 0 and (global_step_t % int(args.step_interval) == 0):
                        step_samples_ep.append(lval)
                        step_losses_flat_t.append(lval)

                    if log_interval_t and ((bi + 1) % log_interval_t == 0):
                        print(f"Torch ep {ep+1}/{args.epochs} batch {bi+1} loss={lval:.4f}")
                per_epoch_batch_losses_t.append(batch_losses_t)
                loss_ep_t = float(np.mean(batch_losses_t)) if batch_losses_t else float('nan')
                per_epoch_losses_t.append(loss_ep_t)
                per_epoch_step_losses_t.append(step_samples_ep)
                print(f"Torch epoch {ep+1}/{args.epochs}: loss={loss_ep_t:.4f} batches={total_batches_t}")
            t_torch = time.time() - t0

            # Final eval (full dataset)
            loss_final, acc_final = evaluate_torch_model(net, ds_t, args.bs)

            summary['torch'] = {
                'loss_before': float(loss_init), 'loss_after': float(loss_final),
                'acc_before': float(acc_init), 'acc_after': float(acc_final),
                'time_s': float(t_torch), 'losses': [float(x) for x in per_epoch_losses_t],
                'batch_losses': [[float(x) for x in bl] for bl in per_epoch_batch_losses_t],
                'step_losses': [[float(x) for x in sl] for sl in per_epoch_step_losses_t],
                'step_losses_flat': [float(x) for x in step_losses_flat_t],
            }
            print(f"Torch model: loss {summary['torch']['loss_before']:.4f}->{summary['torch']['loss_after']:.4f}, acc {summary['torch']['acc_before']:.3f}->{summary['torch']['acc_after']:.3f}, train_time {t_torch:.2f}s")
        except Exception as e:
            print(f"torch not available or failed ({e}); skipping Torch comparison")

    # Plotting
    if args.plot and HAVE_MATPLOTLIB:
        out_png = plot_training_curves(summary, args.epochs, Path(os.getcwd()), plot_deriv=args.plot_deriv)
        if out_png:
            print(f"Saved plot: {out_png}")
    elif args.plot:
        print("matplotlib not available; skipping plotting")

    # Persist results
    out_file = Path(os.getcwd()) / 'results_resnet.txt'
    with open(out_file, 'a') as f:
        f.write(json.dumps(summary) + "\n")
    print(f"Wrote results to {out_file}")


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--n', type=int, default=1024)
    p.add_argument('--bs', type=int, default=64)
    p.add_argument('--epochs', type=int, default=3)
    p.add_argument('--lr', type=float, default=0.05)
    p.add_argument('--momentum', type=float, default=0.9)
    p.add_argument('--plot', action='store_true')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--step-interval', type=int, default=10, help='Record a step loss every k steps (<=0 to disable)')
    p.add_argument('--plot-deriv', action='store_true', help='Also plot improvement (loss delta per step)')
    p.add_argument('--compare-torch', action='store_true', help='Also train and compare a Torch mirror model')
    args = p.parse_args()
    run(args)
