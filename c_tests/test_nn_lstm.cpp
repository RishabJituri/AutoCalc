// ============================
// File: tests/test_nn_lstm.cpp
// ============================
#include "test_framework.hpp"
#include "ag/nn/layers/lstm.hpp"
#include "ag/core/variables.hpp"
#include <vector>
#include <numeric>
#include <cmath>

using ag::Variable;
using ag::nn::LSTMCell;
using ag::nn::LSTM;

static std::vector<float> zeros(std::size_t n) { return std::vector<float>(n, 0.0f); }
static std::vector<float> ones(std::size_t n)  { return std::vector<float>(n, 1.0f); }

TEST("nn/lstm/shape_and_params") {
    const std::size_t I=3, H=4, B=2;
    LSTMCell lstm(I, H, /*bias=*/true, /*init_scale=*/0.01, /*seed=*/777ull);
    auto params = lstm.parameters();
    // 8 weight matrices + 4 biases
    ASSERT_TRUE(params.size() == 12);

    Variable x(zeros(B*I), {B,I}, /*requires_grad=*/true);
    Variable h0(zeros(B*H), {B,H}, /*requires_grad=*/true); // allow grad to test path
    Variable c0(zeros(B*H), {B,H}, /*requires_grad=*/true);
    auto [h1, c1] = lstm.forward_step(x,h0,c0);
    ASSERT_TRUE(h1.shape()[0]==B && h1.shape()[1]==H);
    ASSERT_TRUE(c1.shape()[0]==B && c1.shape()[1]==H);
}

TEST("nn/lstm/numeric_grad_single_step_sum") {
    const std::size_t I=2, H=2, B=1;
    LSTMCell lstm(I, H, /*bias=*/true, /*init_scale=*/0.05, /*seed=*/1234ull);

    // Input and initial states
    std::vector<float> xv = {0.1f, -0.2f};
    Variable x(xv, {B,I}, /*requires_grad=*/true);
    Variable h0(zeros(B*H), {B,H}, /*requires_grad=*/true);
    Variable c0(zeros(B*H), {B,H}, /*requires_grad=*/true);

    auto [h1, c1] = lstm.forward_step(x,h0,c0);
    // Loss = sum(h1)
    float sumv = 0.0f; for (float v : h1.value()) sumv += v;
    h1.backward(ones(h1.value().size()));

    // Finite difference on a couple of x entries
    const float eps = 1e-4f;
    for (std::size_t i=0;i<xv.size();++i) {
        auto xp = xv; xp[i] += eps;
        auto xm = xv; xm[i] -= eps;
        Variable Xp(xp, {B,I}, /*requires_grad=*/false);
        Variable Xm(xm, {B,I}, /*requires_grad=*/false);
        auto hp = lstm.forward_step(Xp,h0,c0).first;
        auto hm = lstm.forward_step(Xm,h0,c0).first;
        float fp = 0.0f, fm = 0.0f;
        for (float v : hp.value()) fp += v;
        for (float v : hm.value()) fm += v;
        float gnum = (fp - fm) / (2*eps);
        ASSERT_NEAR(x.grad()[i], gnum, 1e-3f);
    }
}

TEST("nn/lstm/unrolled_three_steps_determinism") {
    const std::size_t I=3, H=3, B=2, T=3;
    LSTMCell lstm(I, H, /*bias=*/true, /*init_scale=*/0.03, /*seed=*/99ull);

    // sequence x_t
    std::vector<Variable> xs;
    for (std::size_t t=0;t<T;++t) {
        std::vector<float> xv(B*I);
        for (std::size_t i=0;i<xv.size();++i) xv[i] = float(i + 1 + t)*0.1f;
        xs.emplace_back(Variable(xv, {B,I}, /*requires_grad=*/true));
    }
    Variable h(zeros(B*H), {B,H}, /*requires_grad=*/true);
    Variable c(zeros(B*H), {B,H}, /*requires_grad=*/true);

    // forward unroll
    for (std::size_t t=0;t<T;++t) {
        auto hc = lstm.forward_step(xs[t], h, c);
        h = hc.first;
        c = hc.second;
    }
    // Loss = sum of final h
    h.backward(ones(h.value().size()));
    auto g1 = xs[0].grad(); // capture grads

    // zero
    for (auto& xt : xs) xt.zero_grad();
    h.zero_grad(); c.zero_grad();

    // recompute
    Variable h2(zeros(B*H), {B,H}, /*requires_grad=*/true);
    Variable c2(zeros(B*H), {B,H}, /*requires_grad=*/true);
    for (std::size_t t=0;t<T;++t) {
        auto hc = lstm.forward_step(xs[t], h2, c2);
        h2 = hc.first;
        c2 = hc.second;
    }
    h2.backward(ones(h2.value().size()));

    // grads on xs[0] should match (deterministic recompute)
    auto g2 = xs[0].grad();
    ASSERT_TRUE(g1.size() == g2.size());
    for (std::size_t i=0;i<g1.size();++i) ASSERT_NEAR(g1[i], g2[i], 1e-6f);
}

TEST("nn/lstm/shape_forward_and_backward") {
  const std::size_t B = 2, T = 3, I = 4, H = 5;

  // Build X:[B,T,I]
  std::vector<float> x(B * T * I);
  for (std::size_t b = 0; b < B; ++b)
    for (std::size_t t = 0; t < T; ++t)
      for (std::size_t i = 0; i < I; ++i)
        x[(b*T + t)*I + i] = 0.01f * float((b+1)*(t+1)*(i+1));

  Variable X(x, {B, T, I}, /*requires_grad=*/false);

  LSTM lstm(I, H, /*num_layers=*/1, /*bias=*/true);
  auto Y = lstm.forward(X);           // expect [B,T,H]

  auto sh = Y.shape();
  ASSERT_TRUE(sh.size() == 3);
  ASSERT_TRUE(sh[0] == B);
  ASSERT_TRUE(sh[1] == T);
  ASSERT_TRUE(sh[2] == H);

  const auto& yv = Y.value();
  ASSERT_TRUE(yv.size() == B * T * H);

  // Backprop sanity: seed with ones and ensure it runs
  std::vector<float> seed(yv.size(), 1.0f);
  Y.backward(seed);   // passes if no throw/crash
}

TEST("nn/lstm/input_size_mismatch_throws") {
  const std::size_t B = 1, T = 2, I = 3, H = 4;

  // Wrong last dim (I+1) to provoke a clear error
  std::vector<float> x(B * T * (I + 1), 0.0f);
  Variable X_bad(x, {B, T, I + 1}, /*requires_grad=*/false);

  LSTM lstm(I, H, /*num_layers=*/1, /*bias=*/true);

  bool threw = false;
  try {
    (void)lstm.forward(X_bad);
  } catch (...) {
    threw = true;
  }
  ASSERT_TRUE(threw);
}

TEST("nn/lstm/multilayer_shape") {
  const std::size_t B = 2, T = 4, I = 3, H = 6;

  std::vector<float> x(B * T * I, 0.0f);
  Variable X(x, {B, T, I}, /*requires_grad=*/false);

  LSTM lstm(I, H, /*num_layers=*/2, /*bias=*/true);
  auto Y = lstm.forward(X);

  auto sh = Y.shape();
  ASSERT_TRUE(sh.size() == 3);
  ASSERT_TRUE(sh[0] == B);
  ASSERT_TRUE(sh[1] == T);
  ASSERT_TRUE(sh[2] == H);
}

