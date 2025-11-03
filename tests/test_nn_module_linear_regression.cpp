#include "test_framework.hpp"
#include "ag/core/variables.hpp"
#include "ag/nn/layers/linear.hpp"
#include "ag/nn/optim/sgd.hpp"
#include "ag/ops/elementwise.hpp"
#include "ag/ops/reduce.hpp"
#include <random>
#include <vector>
#include <cstddef>
#include <cmath>

using ag::Variable;
using ag::nn::Linear;
using ag::nn::SGD;
using ag::reduce_mean;
using ag::sub;
using ag::mul;

// Small Module wrapper around a single Linear layer (in->1)
struct LinearRegNet : ag::nn::Module {
  Linear fc{2, 1, /*bias=*/true, /*init_scale=*/0.1f, /*seed=*/12345ull};
  LinearRegNet() { register_module("fc", fc); }
  Variable forward(const Variable& x) override { return fc.forward(x); }
protected:
  std::vector<ag::Variable*> _parameters() override { return {}; }
};

static void make_linear_dataset(std::size_t B,
                                std::vector<float>& X,
                                std::vector<float>& y) {
  std::mt19937 rng(1337);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  X.resize(B * 2);
  y.resize(B * 1);
  // ground-truth linear mapping: y = 1.5*x0 - 2.0*x1 + 0.3
  for (std::size_t i = 0; i < B; ++i) {
    float x0 = dist(rng);
    float x1 = dist(rng);
    X[2*i + 0] = x0;
    X[2*i + 1] = x1;
    y[i + 0] = 1.5f * x0 - 2.0f * x1 + 0.3f;
  }
}

TEST("nn/module_linear_regression_converges") {
  const std::size_t B = 128;
  std::vector<float> xv; std::vector<float> yv;
  make_linear_dataset(B, xv, yv);

  LinearRegNet net;
  SGD opt(/*lr=*/0.1f, /*momentum=*/0.0f, /*nesterov=*/false, /*weight_decay=*/0.0f);

  Variable X(xv, {B, 2}, /*requires_grad=*/true);
  Variable Y(yv, {B, 1}, /*requires_grad=*/false);

  // initial loss (MSE mean over batch and output dim)
  auto Yhat0 = net.forward(X);
  auto diff0 = sub(Yhat0, Y);
  auto L0 = reduce_mean(mul(diff0, diff0), /*axes=*/{0,1}, /*keepdims=*/false);
  float loss0 = 0.0f; for (float v : L0.value()) loss0 += v;

  // Train
  for (int step = 0; step < 300; ++step) {
    auto Yhat = net.forward(X);
    auto diff = sub(Yhat, Y);
    auto L = reduce_mean(mul(diff, diff), /*axes=*/{0,1}, /*keepdims=*/false);
    L.backward();
    opt.step(net);
    net.zero_grad();
  }

  auto Yhat1 = net.forward(X);
  auto diff1 = sub(Yhat1, Y);
  auto L1 = reduce_mean(mul(diff1, diff1), /*axes=*/{0,1}, /*keepdims=*/false);
  float loss1 = 0.0f; for (float v : L1.value()) loss1 += v;

  // Expect >10x reduction
  ASSERT_TRUE(loss1 < loss0 * 0.1f);
}
