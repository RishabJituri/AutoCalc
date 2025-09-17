// tests/test_nn_lstm_layer.cpp
#include "test_framework.hpp"
#include "ag/nn/layers/lstm.hpp"   // sequence LSTM appended to your cell files
#include "ag/core/variables.hpp"
#include <vector>
#include <stdexcept>
#include <cmath>

using ag::Variable;
using ag::nn::LSTM;

TEST("nn/lstm/shape_forward_and_backward") {
  const std::size_t B = 2, T = 3, I = 4, H = 5;

  // Build X:[B,T,I]
  std::vector<float> x(B * T * I);
  for (std::size_t b = 0; b < B; ++b)
    for (std::size_t t = 0; t < T; ++t)
      for (std::size_t i = 0; i < I; ++i)
        x[(b*T + t)*I + i] = 0.01 * float((b+1)*(t+1)*(i+1));

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
  std::vector<float> seed(yv.size(), 1.0);
  Y.backward(seed);   // passes if no throw/crash
}

TEST("nn/lstm/input_size_mismatch_throws") {
  const std::size_t B = 1, T = 2, I = 3, H = 4;

  // Wrong last dim (I+1) to provoke a clear error
  std::vector<float> x(B * T * (I + 1), 0.0);
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

  std::vector<float> x(B * T * I, 0.0);
  Variable X(x, {B, T, I}, /*requires_grad=*/false);

  LSTM lstm(I, H, /*num_layers=*/2, /*bias=*/true);
  auto Y = lstm.forward(X);

  auto sh = Y.shape();
  ASSERT_TRUE(sh.size() == 3);
  ASSERT_TRUE(sh[0] == B);
  ASSERT_TRUE(sh[1] == T);
  ASSERT_TRUE(sh[2] == H);
}
