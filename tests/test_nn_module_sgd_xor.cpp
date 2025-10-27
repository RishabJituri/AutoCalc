// filepath: tests/test_nn_module_sgd_xor.cpp
#include "test_framework.hpp"
#include "ag/core/variables.hpp"
#include "ag/nn/layers/linear.hpp"
#include "ag/nn/loss.hpp"
#include "ag/nn/optim/sgd.hpp"
#include "ag/ops/activations.hpp"
#include <vector>
#include <cstddef>
#include <cmath>

using ag::Variable;
using ag::nn::Linear;
using ag::nn::SGD;

// Two-layer MLP for XOR: 2 -> 8 -> 2
struct XorNet : ag::nn::Module {
  Linear l1{2, 8, /*bias=*/true, /*init_scale=*/0.1f, /*seed=*/321ull};
  Linear l2{8, 2, /*bias=*/true, /*init_scale=*/0.1f, /*seed=*/654ull};
  XorNet() { register_module("l1", l1); register_module("l2", l2); }
  Variable forward(const Variable& x) override {
    auto h = ag::relu(l1.forward(x));
    return l2.forward(h);
  }
protected:
  std::vector<ag::Variable*> _parameters() override { return {}; }
};

TEST("nn/module_sgd_xor_converges") {
  // XOR dataset repeated (batch learning on entire set)
  const std::size_t B = 4;
  std::vector<float> X = {0,0,  0,1,  1,0,  1,1};
  std::vector<std::size_t> y = {0, 1, 1, 0};
  Variable Xv(X, {B, 2}, /*requires_grad=*/true);

  XorNet net;
  SGD opt(/*lr=*/0.5f, /*momentum=*/0.0f, /*nesterov=*/false, /*weight_decay=*/0.0f);

  // Train for some epochs on the tiny dataset
  for (int epoch = 0; epoch < 2000; ++epoch) {
    auto logits = net.forward(Xv);
    auto loss = ag::nn::cross_entropy(logits, y);
    loss.backward();
    opt.step(net);
  }

  // Check predictions
  auto logits = net.forward(Xv);
  const auto& z = logits.value();
  std::size_t correct = 0;
  for (std::size_t i = 0; i < B; ++i) {
    std::size_t pred = (z[2*i+1] > z[2*i+0]) ? 1u : 0u;
    if (pred == y[i]) ++correct;
  }
  float acc = float(correct)/float(B);
  ASSERT_GE(acc, 0.99f);
}
