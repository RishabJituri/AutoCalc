# AutoCalc — C++ and Python APIs

This repository provides a small neural-network/autograd engine in C++ with Python bindings.
This README now documents both:
- The Python bindings (quickstart, API overview, and MNIST benchmarks)
- The C++ API (unchanged and still available below)

---

## Quick links

- Python quickstart: Python bindings setup and smoke tests
- Python API overview: ag.Variable, ops, ag.nn (layers/optim/losses), ag.data (Dataset/DataLoader)
- Benchmarks (Python): `py_demos/ResNet_MNIST.py`, `py_demos/SimpleNet_MNIST.py`
- C++ headers: `include/ag/`
- C demos (C++): `c_demos/`
- Tests (C++): `tests/`

---

## Python bindings — quickstart (macOS, zsh)

1) Build the project (produces the Python package under `build/python/ag`)

```zsh
mkdir -p build
cd build
cmake ..
cmake --build . -- -j
```

2) Point Python at the built bindings (no pip install required)

```zsh
# From repo root
export PYTHONPATH="$(pwd)/build/python:$PYTHONPATH"
```

3) Smoke test

```zsh
python - <<'PY'
import ag, numpy as np
x = ag.Variable.from_numpy(np.ones((2,3), dtype=np.float32), requires_grad=True)
y = ag.relu(x)
y_sum = np.asarray(y.value()).sum()
print('OK ag:', y_sum)
PY
```

If this prints `OK ag: ...`, the bindings are importable and ops work end-to-end.

Data location note: Python benchmarks/scripts expect local MNIST IDX files under `Data/MNIST/raw` (train-images-idx3-ubyte, train-labels-idx1-ubyte, etc.). No torchvision is required.

---

## Python API overview

The Python package roughly mirrors the C++ API and exposes a lightweight, PyTorch-like surface:

- Tensors/autograd: `ag.Variable`
  - Construct from numpy: `ag.Variable.from_numpy(np_array, requires_grad=...)`
  - Common ops: `ag.relu`, `ag.flatten`, `ag.reshape`, elementwise arithmetic
  - Inspect: `v.shape()`; convert: `np.asarray(v.value())`

- Neural network: `ag.nn`
  - Modules: subclass `ag.nn.Module` and implement `forward(self, x)`
  - Layers: `Conv2d`, `BatchNorm2d`, `Linear`, `AvgPool2d`, `Dropout`, etc.
  - Optimizers: `ag.nn.SGD(lr, momentum)`; call `model.zero_grad()`, `loss.backward()`, `opt.step(model)`
  - Losses: `ag.nn.cross_entropy(logits, targets)`

- Data: `ag.data`
  - Define a dataset by subclassing `ag.data.Dataset` with `size()` and `get(idx)` returning an `ag.data.Example` with `x`, `y`.
  - Batch with `ag.data.DataLoader(dataset, options)` where `options = ag.data.DataLoaderOptions()` (set `batch_size`, `shuffle`, `seed`).

Minimal dataset example (MNIST IDX without torchvision):

```python
import os, struct, numpy as np, ag

class LocalMNIST(ag.data.Dataset):
    def __init__(self, root, train=True, max_samples=None):
        super().__init__()
        d = os.path.join(root, 'MNIST', 'raw')
        img = os.path.join(d, 'train-images-idx3-ubyte' if train else 't10k-images-idx3-ubyte')
        lbl = os.path.join(d, 'train-labels-idx1-ubyte' if train else 't10k-labels-idx1-ubyte')
        with open(img,'rb') as f:
            _, n, r, c = struct.unpack('>IIII', f.read(16))
            X = np.frombuffer(f.read(), dtype=np.uint8).reshape(n, r, c)
        with open(lbl,'rb') as f:
            _, n2 = struct.unpack('>II', f.read(8))
            Y = np.frombuffer(f.read(), dtype=np.uint8)
        if max_samples:
            X, Y = X[:max_samples], Y[:max_samples]
        self.X, self.Y = X, Y
    def size(self):
        return int(self.X.shape[0])
    def get(self, idx):
        i = int(idx)
        x = (self.X[i].astype(np.float32)/255.0)[None, :, :]
        y = np.array([int(self.Y[i])], dtype=np.int64)
        ex = ag.data.Example()
        ex.x = ag.Variable.from_numpy(x, requires_grad=False)
        ex.y = ag.Variable.from_numpy(y, requires_grad=False)
        return ex
```

Training sketch (SmallResNet):

```python
class SmallResBlock(ag.nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.conv1 = ag.nn.Conv2d(ch, ch, 3,3,1,1,1,1, bias=True, init_scale=0.02, seed=0)
        self.bn1 = ag.nn.BatchNorm2d(ch)
        self.conv2 = ag.nn.Conv2d(ch, ch, 3,3,1,1,1,1, bias=True, init_scale=0.02, seed=1)
        self.bn2 = ag.nn.BatchNorm2d(ch)
    def forward(self, x):
        id = x
        x = ag.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x)) + id
        return ag.relu(x)

class SmallResNet(ag.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = ag.nn.Conv2d(1, 8, 3,3,1,1,1,1, bias=True, init_scale=0.02, seed=2)
        self.bn = ag.nn.BatchNorm2d(8)
        self.block = SmallResBlock(8)
        self.pool = ag.nn.AvgPool2d(28,28,0,0,0,0)
        self.fc = ag.nn.Linear(8, 10, seed=3)
    def forward(self, x):
        x = ag.relu(self.bn(self.conv(x)))
        x = self.block(x)
        x = self.pool(x)
        x = ag.flatten(x, 1)
        return self.fc(x)

opt = ag.nn.SGD(0.05, 0.9)
```

---

## Python MNIST benchmarks

Two ready-to-run scripts demonstrate the Python API and produce plots/summaries to compare training behavior.

Common setup (from repo root):

```zsh
export PYTHONPATH="$(pwd)/build/python:$PYTHONPATH"
```

- SimpleNet (AG only, optional Torch compare):

```zsh
python py_demos/SimpleNet_MNIST.py --n 1024 --bs 64 --epochs 3 --plot
```

- ResNet (AG, optional Torch mirror + combined plots):

```zsh
python py_demos/ResNet_MNIST.py \
  --n 35000 --bs 64 --epochs 1 \
  --seed 0 --plot --step-interval 10 --plot-deriv \
  --compare-torch
```

What the scripts do:
- Load MNIST IDX directly (no torchvision) from `Data/MNIST/raw`
- Build identical SmallResNet in AG and Torch
- Train with per-epoch, per-batch, and step-sampled loss tracking
- Optionally compute and plot “improvement” (Δloss) per step
- Save plots (Agg backend) and append JSON lines to `results_resnet.txt`

Artifacts:
- `results_resnet_*.png` (epoch loss)
- `results_resnet_train_steps_*.png` (per-batch loss, combined)
- `results_resnet_train_steps_sampled_*.png` (every-k-step loss, combined)
- `results_resnet_train_steps_improvement_*.png` and `*_sampled_improvement_*.png` when `--plot-deriv` is set
- `results_resnet.txt` summaries (AG and Torch blocks)

Troubleshooting Torch accuracy:
- BatchNorm running stats can underperform on very few batches. Try more epochs, higher BN momentum, or evaluate in mini-batches using train-mode stats.

---

## C++ API (original documentation)

This repository provides a small neural-network/autograd engine in C++.
This README documents the public C++ API surface that is most useful to C++ consumers: the `Variable` autograd type and the high-level neural-network components (`Module`, `Sequential`, common layers and optimizers).

This file is a concise developer reference — see the headers under `include/ag/` for full signatures and the `tests/` and `c_demos/` directories for usage patterns.

---

## Quick links

- Source headers: `include/ag/`
- Core autograd: `include/ag/core/variables.hpp`
- Ops: `include/ag/ops/`
- NN: `include/ag/nn/`
- Data utilities: `include/ag/data/dataloader.hpp`
- C demos: `c_demos/`
- Tests: `tests/`

---

## Build (macOS, zsh)

Create a build directory and build the default targets (tests + demos where available):

```zsh
mkdir -p build
cd build
cmake ..
cmake --build . -- -j
```

Run the test executable produced by the CMake project:

```zsh
./bin/tests
```

Notes:
- Python bindings (optional) require `pybind11` and Python development headers — the cmake target will be skipped if these are not available.
- Sanitizer flags are configured in `CMakeLists.txt`; enabling them on macOS requires a compatible Clang toolchain.

---

## Include

To use the library in your project include the umbrella header:

```cpp
#include "ag/all.hpp"
using namespace ag;
```

This header pulls in the core autograd `Variable`, ops, and the NN helpers.

---

## Autograd: Variable (overview)

Header: `include/ag/core/variables.hpp`

Purpose: `Variable` is the primary tensor + autograd handle. It wraps a `Node` (value buffer, grad buffer, shape and backward function) and exposes a compact API for computation and differentiation.

Primary constructors
- `Variable()` — empty placeholder Variable.
- `Variable(const std::vector<float>& value, const std::vector<std::size_t>& shape, bool requires_grad = true)` — create a tensor from contiguous float data.
- `explicit Variable(std::shared_ptr<Node> node)` — advanced: wrap an existing Node.

Key methods
- `const std::vector<float>& value() const` — flattened data buffer.
- `const std::vector<float>& grad() const` — flattened gradient buffer.
- `const std::vector<std::size_t>& shape() const` — tensor shape (per-dimension sizes).
- `bool requires_grad() const` — whether gradients will be tracked.
- `void zero_grad()` — set gradients to zero across the reachable subgraph of parameters.
- `void backward()` / `void backward(const std::vector<float>& seed)` — compute gradients with an implicit scalar seed or an explicit tensor seed.

Ops
The project exposes common arithmetic and tensor ops in `include/ag/ops/` and declared in `include/ag/core/variables.hpp`:
- Arithmetic: `add, sub, mul, div, neg, expv, pow, sinv, cosv, ...`
- Linear/batched matmul: `matmul`
- Reshape/transpose/slicing: `transpose`, `at`

Grad mode
- Global thread-local grad mode: `ag::is_grad_enabled()` and `ag::set_grad_enabled(bool)` to control whether newly created Variables require grad.

Advanced
- `Variable::n` is the underlying `std::shared_ptr<Node>` and may be used when implementing custom ops.

---

## Neural network API (overview)

Umbrella headers: `include/ag/nn/*` (included by `ag/all.hpp`).

Key types
- `ag::nn::Module` — base class for layers and models. Implements parameter registration and a `forward` interface.
- `ag::nn::Sequential` — convenience container for ordered modules (common model construction style).
- Layers: `Linear`, `Conv2d`, normalization layers (`BatchNorm2d`, `LayerNorm`), pooling, `Dropout`, `LSTM`.
- Losses: see `include/ag/nn/loss.hpp` (e.g. `cross_entropy` variants).
- Optimizers: e.g. `ag::nn::optim::SGD` in `include/ag/nn/optim/sgd.hpp`.

Typical workflow
1. Construct model (Module / Sequential).
2. Prepare optimizer with `model.parameters()`.
3. For each batch: forward, compute loss, zero grads, backward, optimizer step.

Example (sketch):

```cpp
using namespace ag;
using namespace ag::nn;

Sequential net({
  std::make_shared<Linear>(784, 128),
  std::make_shared<ReLU>(),
  std::make_shared<Linear>(128, 10)
});

auto params = net.parameters();
ag::nn::optim::SGD sgd(params, /*lr=*/0.01f, /*momentum=*/0.9f);

Variable x = /* batch input */;
Variable y = /* labels */;

Variable logits = net.forward(x);
Variable loss = cross_entropy(logits, y);

for (auto &p : params) p.zero_grad();
loss.backward();
sgd.step();
```

See `tests/test_nn_*` for concrete end-to-end examples and regression tests demonstrating common setups (XOR, spiral, ResNet linear head, LSTM tests, etc.).

---

## DataLoader

A simple `DataLoader` implementation is available under `include/ag/data/dataloader.hpp` and tests in `tests/test_data_dataloader*.cpp`.
Use `InMemoryDataset` for easy experiments and `DataLoaderOptions` to configure batch size, shuffling, num workers, and prefetch.

---

## C demos & experiments

- `c_demos/mnist_demo.cpp` — training loop and dataset loading for MNIST/FashionMNIST.
- `experiments/` — small Python experiments and scripts used during development.

Recorded runs and metrics are stored in `results_mnist.txt` and `results_resnet.txt`.
