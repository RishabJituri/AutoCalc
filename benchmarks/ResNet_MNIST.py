#!/usr/bin/env python3
"""
Simple benchmark script to compare the local SmallResNet (ag) against a torchvision ResNet18
on the local MNIST raw files. Writes a brief summary to results_resnet.txt.

Usage:
  python resnet_mnist/run_benchmark.py --n 1024 --bs 64 --epochs 3 --compare-torch

The script is robust to missing torchvision/torch: it will run the ag model and skip the torch
comparison if torch is not installed.
"""
import os
import time
import argparse
import json
from pathlib import Path

import numpy as np
import ag

# Minimal IDX reader (no torchvision dependency)
import struct
import datetime

try:
    import matplotlib.pyplot as plt
    HAVE_MATPLOTLIB = True
except Exception:
    HAVE_MATPLOTLIB = False

def load_mnist_idx(root, train=True, max_samples=None):
    mnist_raw_dir = os.path.join(root, "MNIST", "raw")
    if train:
        img_path = os.path.join(mnist_raw_dir, "train-images-idx3-ubyte")
        lbl_path = os.path.join(mnist_raw_dir, "train-labels-idx1-ubyte")
    else:
        img_path = os.path.join(mnist_raw_dir, "t10k-images-idx3-ubyte")
        lbl_path = os.path.join(mnist_raw_dir, "t10k-labels-idx1-ubyte")
    if not (os.path.exists(img_path) and os.path.exists(lbl_path)):
        raise FileNotFoundError(f"MNIST idx files not found under {mnist_raw_dir}")

    with open(img_path, 'rb') as f:
        magic, num, rows, cols = struct.unpack('>IIII', f.read(16))
        data = np.frombuffer(f.read(), dtype=np.uint8)
        data = data.reshape(num, rows, cols)
    with open(lbl_path, 'rb') as f:
        magic_l, num_l = struct.unpack('>II', f.read(8))
        labels = np.frombuffer(f.read(), dtype=np.uint8)

    if max_samples:
        take = min(max_samples, num)
        data = data[:take]
        labels = labels[:take]
    # expand channel dim
    data = data.astype(np.float32) / 255.0
    data = data[:, None, :, :]
    labels = labels.astype(np.int64)
    return data, labels


# Define small ResNet compatible with ag bindings
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
        self.pool = ag.nn.AvgPool2d(28,28,0,0,0,0)  # global avg pool for 28x28
        self.fc = ag.nn.Linear(8, num_classes, seed=3)
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = ag.relu(x)
        x = self.block1(x)
        x = self.pool(x)
        x = ag.flatten(x, 1)
        x = self.fc(x)
        return x


def var_to_numpy(v):
    try:
        if callable(getattr(v, "value", None)):
            return np.asarray(v.value())
    except Exception:
        pass
    try:
        return np.asarray(v)
    except Exception:
        return None


def evaluate_ag_model(model, X, Y, batch_size):
    # ag binding: set eval mode
    try:
        model.train(False)
    except Exception:
        pass
    n = X.shape[0]
    losses = []
    correct = 0
    for i in range(0, n, batch_size):
        xb = X[i:i+batch_size]
        yb = Y[i:i+batch_size]
        b = xb.shape[0]
        x_var = ag.Variable.from_numpy(xb)
        logits = model(x_var)
        targets = list(map(int, yb.tolist()))
        L = ag.nn.cross_entropy(logits, targets)
        # get loss scalar robustly
        L_np = var_to_numpy(L)
        try:
            if hasattr(L_np, 'item'):
                loss_scalar = float(L_np.item())
            else:
                loss_scalar = float(np.asarray(L_np))
        except Exception:
            loss_scalar = float(np.asarray(L_np).mean())
        losses.append(loss_scalar)

        logits_np = var_to_numpy(logits)
        if logits_np is None:
            raise RuntimeError('logits conversion to numpy failed')
        # reshape flattened logits if necessary
        if logits_np.ndim == 1:
            # infer classes
            if logits_np.size % b == 0:
                k = logits_np.size // b
                logits_np = logits_np.reshape(b, k)
        elif logits_np.ndim == 2 and logits_np.shape[0] == 1 and b > 1:
            # sometimes shape confusion: ensure first dim equals batch size
            if logits_np.size == b * logits_np.shape[1]:
                logits_np = logits_np.reshape(b, logits_np.shape[1])

        pred = np.argmax(logits_np, axis=1)
        correct += int((pred == yb).sum())
    return float(np.mean(losses)), correct / float(n)


def train_ag(model, X, Y, batch_size, epochs, lr, momentum, eval_after_epoch=True):
    opt = ag.nn.SGD(lr, momentum)
    n = X.shape[0]
    per_epoch_losses = []
    start = time.time()
    try:
        model.train(True)
    except Exception:
        pass
    for ep in range(epochs):
        # simple shuffle
        idx = np.random.permutation(n)
        for i in range(0, n, batch_size):
            ids = idx[i:i+batch_size]
            xb = X[ids]
            yb = Y[ids]
            x_var = ag.Variable.from_numpy(xb)
            targets = list(map(int, yb.tolist()))
            model.zero_grad()
            logits = model(x_var)
            L = ag.nn.cross_entropy(logits, targets)
            L.backward()
            opt.step(model)
        if eval_after_epoch:
            loss_ep, acc_ep = evaluate_ag_model(model, X, Y, batch_size)
            per_epoch_losses.append(loss_ep)
            print(f"AG epoch {ep+1}/{epochs}: loss={loss_ep:.4f}, acc={acc_ep:.3f}")
    duration = time.time() - start
    return duration, per_epoch_losses


def try_import_torch():
    try:
        import torch
        import torchvision
        return True
    except Exception:
        return False


def run(args):
    root = os.path.abspath(os.path.join(os.getcwd(), "Data"))
    X, Y = load_mnist_idx(root, train=True, max_samples=args.n)
    print(f"Loaded {X.shape[0]} samples")

    # Seed control
    np.random.seed(args.seed)
    model_ag = SmallResNet(num_classes=10)
    loss0, acc0 = evaluate_ag_model(model_ag, X, Y, args.bs)
    t_ag, ag_losses = train_ag(model_ag, X, Y, args.bs, args.epochs, args.lr, args.momentum)
    loss1, acc1 = evaluate_ag_model(model_ag, X, Y, args.bs)
    print(f"AG model: loss {loss0:.4f}->{loss1:.4f}, acc {acc0:.3f}->{acc1:.3f}, train_time {t_ag:.2f}s")

    summary = {
        'n': int(args.n), 'bs': int(args.bs), 'epochs': int(args.epochs),
        'ag': {'loss_before': loss0, 'loss_after': loss1, 'acc_before': acc0, 'acc_after': acc1, 'time_s': t_ag, 'losses': ag_losses}
    }

    if args.compare_torch and try_import_torch():
        import torch
        import torch.nn as nn
        import torch.optim as optim

        # Build a Torch model matching SmallResNet topology for fair comparison
        torch.manual_seed(args.seed)
        init_scale = 0.02

        class TorchSmallBlock(nn.Module):
            def __init__(self, channels):
                super().__init__()
                self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, bias=True)
                self.bn1 = nn.BatchNorm2d(channels)
                self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, bias=True)
                self.bn2 = nn.BatchNorm2d(channels)
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
                self.block1 = TorchSmallBlock(8)
                self.pool = nn.AvgPool2d(28)
                self.fc = nn.Linear(8, num_classes)
                # initialize weights to match AG init_scale
                for m in self.modules():
                    if isinstance(m, (nn.Conv2d, nn.Linear)):
                        nn.init.normal_(m.weight, mean=0.0, std=init_scale)
                        if m.bias is not None:
                            nn.init.zeros_(m.bias)
                    elif isinstance(m, nn.BatchNorm2d):
                        nn.init.ones_(m.weight)
                        nn.init.zeros_(m.bias)

            def forward(self, x):
                x = self.conv(x)
                x = self.bn(x)
                x = nn.functional.relu(x)
                x = self.block1(x)
                x = self.pool(x)
                # flatten keeping batch dim
                x = torch.flatten(x, 1)
                x = self.fc(x)
                return x

        torch_resnet = TorchSmallResNet(num_classes=10)
        torch_resnet.train()

        # prepare torch tensors
        X_t = torch.from_numpy(X).float()
        Y_t = torch.from_numpy(Y).long().squeeze()
        opt_t = optim.SGD(torch_resnet.parameters(), lr=args.lr, momentum=args.momentum)
        criterion = nn.CrossEntropyLoss()

        t_start = time.time()
        torch_losses = []
        for ep in range(args.epochs):
            perm = torch.randperm(X_t.shape[0])
            for i in range(0, X_t.shape[0], args.bs):
                ids = perm[i:i+args.bs]
                xb = X_t[ids]
                yb = Y_t[ids]
                opt_t.zero_grad()
                out = torch_resnet(xb)
                loss = criterion(out, yb)
                loss.backward()
                opt_t.step()
            # eval after epoch
            torch_resnet.eval()
            with torch.no_grad():
                out = torch_resnet(X_t)
                loss_ep = float(criterion(out, Y_t).item())
                pred = out.argmax(dim=1).numpy()
                acc_ep = float((pred == Y_t.numpy()).mean())
                torch_losses.append(loss_ep)
                print(f"Torch epoch {ep+1}/{args.epochs}: loss={loss_ep:.4f}, acc={acc_ep:.3f}")
            torch_resnet.train()
        t_torch = time.time() - t_start

        summary['torch'] = {'time_s': t_torch, 'losses': torch_losses}
        print(f"Torch small ResNet: train_time {t_torch:.2f}s")

    else:
        if args.compare_torch:
            print("torch/torchvision not available; skipping Torch comparison")

    # Optionally plot losses
    if args.plot:
        if not HAVE_MATPLOTLIB:
            print("matplotlib not available; skipping plotting")
        else:
            plt.figure()
            epochs_axis = list(range(1, args.epochs+1))
            if 'ag' in summary and 'losses' in summary['ag']:
                plt.plot(epochs_axis, summary['ag']['losses'], label='ag')
            if 'torch' in summary and 'losses' in summary['torch']:
                plt.plot(epochs_axis, summary['torch']['losses'], label='torch')
            plt.xlabel('Epoch')
            plt.ylabel('CrossEntropy Loss')
            plt.legend()
            out_png = Path(os.getcwd()) / f"results_resnet_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            plt.savefig(out_png)
            print(f"Saved loss plot to {out_png}")

    # Append summary to results_resnet.txt
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
    p.add_argument('--compare-torch', action='store_true')
    p.add_argument('--plot', action='store_true', help='Save a loss plot comparing ag and torch')
    p.add_argument('--seed', type=int, default=0, help='Random seed for reproducibility')
    args = p.parse_args()
    run(args)
