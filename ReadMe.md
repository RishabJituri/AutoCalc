# AutoCalc — C++ API

This repository provides a small neural-network/autograd engine in C++.
This README documents the public C++ API surface that is most useful to C++ consumers: the `Variable` autograd type and the high-level neural-network components (`Module`, `Sequential`, common layers and optimizers).

This file is a concise developer reference — see the headers under `include/ag/` for full signatures and the `tests/` and `examples/` directories for usage patterns.

---

## Quick links

- Source headers: `include/ag/`
- Core autograd: `include/ag/core/variables.hpp`
- Ops: `include/ag/ops/`
- NN: `include/ag/nn/`
- Data utilities: `include/ag/data/dataloader.hpp`
- Examples: `examples/`
- Tests: `tests/`

---

## Build (macOS, zsh)

Create a build directory and build the default targets (tests + examples where available):

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

## Examples & experiments

- `examples/mnist_demo.cpp` — training loop and dataset loading for MNIST/FashionMNIST.
- `experiments/` — small Python experiments and scripts used during development.

Recorded runs and metrics are stored in `results_mnist.txt` and `results_resnet.txt`.

---

## Contributing

- Add new ops under `include/ag/ops` and `src/ag/ops` and follow the patterns used by existing ops.
- Add new layers under `include/ag/nn/layers` and `src/ag/nn/layers`.
- Add tests to `tests/` and run `./bin/tests` frequently.

---

## License

Add your project license here (e.g. MIT, Apache-2.0). Update this file to reflect the chosen license.

---

If you want this file to include full API signatures (all public methods/types), or want a separate `README_API.md` focused purely on reference, say which format you prefer and I will add it.