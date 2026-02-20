// filepath: tests/test_nn_module_sgd_convergence.cpp
#include "test_framework.hpp"
#include "ag/core/variables.hpp"
#include "ag/nn/layers/linear.hpp"
#include "ag/nn/loss.hpp"
#include "ag/nn/optim/sgd.hpp"
#include <random>
#include <vector>
#include <cstddef>
#include <cmath>

using ag::Variable;
using ag::nn::Linear;
using ag::nn::SGD;

// Simple classifier Module wrapping a Linear layer.
struct SimpleNet : ag::nn::Module {
  Linear fc{2, 2, /*bias=*/true, /*init_scale=*/0.01f, /*seed=*/123ull};
  SimpleNet() { register_module("fc", fc); }
  Variable forward(const Variable& x) override { return fc.forward(x); }
protected:
  std::vector<ag::Variable*> _parameters() override { return {}; }
};

static void make_linear_plane_dataset(std::size_t B,
                                      std::vector<float>& X,
                                      std::vector<std::size_t>& y) {
  std::mt19937 rng(42);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  X.resize(B * 2);
  y.resize(B);
  // True separating plane: 0.7*x0 - 0.3*x1 + 0.1
  for (std::size_t i = 0; i < B; ++i) {
    float x0 = dist(rng);
    float x1 = dist(rng);
    X[2*i + 0] = x0;
    X[2*i + 1] = x1;
    float s = 0.7f * x0 - 0.3f * x1 + 0.1f;
    y[i] = (s >= 0.0f) ? 1u : 0u;
  }
}

TEST("nn/module_sgd_linear_classification_converges") {
  // Generate a fixed dataset
  const std::size_t B = 512;
  std::vector<float> xv; std::vector<std::size_t> yv;
  make_linear_plane_dataset(B, xv, yv);

  // Model and optimizer
  SimpleNet net;
  SGD opt(/*lr=*/0.5f, /*momentum=*/0.0f, /*nesterov=*/false, /*weight_decay=*/0.0f);

  // Training tensor
  Variable X(xv, {B, 2}, /*requires_grad=*/true);

  // Initial loss (not asserted, just for sanity)
  auto logits0 = net.forward(X);
  auto L0 = ag::nn::cross_entropy(logits0, yv);
  (void)L0;

  // Train for a number of steps on the same batch
  for (int step = 0; step < 200; ++step) {
    auto logits = net.forward(X);
    auto loss = ag::nn::cross_entropy(logits, yv);
    loss.backward();
    opt.step(net);
    // net.zero_grad(); // not necessary if optimizer zeroes grads; kept optional
  }

  // Evaluate accuracy
  auto logits = net.forward(X);
  const auto& z = logits.value();
  std::size_t correct = 0;
  for (std::size_t i = 0; i < B; ++i) {
    float z0 = z[2*i + 0];
    float z1 = z[2*i + 1];
    std::size_t pred = (z1 > z0) ? 1u : 0u;
    if (pred == yv[i]) ++correct;
  }
  float acc = float(correct) / float(B);
  ASSERT_GE(acc, 0.98f);
}
