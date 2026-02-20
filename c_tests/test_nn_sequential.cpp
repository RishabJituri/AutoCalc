#include "test_framework.hpp"
#include "ag/nn/sequential.hpp"
#include "ag/nn/layers/linear.hpp"
#include "ag/core/variables.hpp"
#include <memory>
#include <vector>
#include <cmath>

using ag::Variable;
using ag::nn::Sequential;
using ag::nn::Linear;

TEST("nn/sequential/forward_backward_and_params") {
  const std::size_t B = 4;
  const std::size_t IN = 8;
  const std::size_t MID = 16;
  const std::size_t OUT = 8;

  Sequential seq;
  // push two Linear layers
  seq.push(std::make_shared<Linear>(IN, MID, true, 0.02f, 0xC0FFEEULL));
  seq.push(std::make_shared<Linear>(MID, OUT, true, 0.02f, 0xD00DULL));

  // random-ish input
  std::vector<float> xv(B * IN);
  for (std::size_t i = 0; i < xv.size(); ++i) xv[i] = float((int(i) % 7) - 3) * 0.1f;
  Variable X(xv, {B, IN}, /*requires_grad=*/true);

  auto Y = seq.forward(X);
  auto sh = Y.shape();
  ASSERT_TRUE(sh.size() == 2);
  ASSERT_TRUE(sh[0] == B && sh[1] == OUT);

  // Backprop seed ones
  std::vector<float> seed(Y.value().size(), 1.0f);
  Y.backward(seed);

  // Ensure parameters exist and at least one param has a non-zero grad
  auto params = seq.parameters();
  ASSERT_TRUE(params.size() > 0);
  bool any_grad = false;
  for (auto pptr : params) {
    if (!pptr) continue;
    const auto& g = pptr->grad();
    if (g.size() == 0) continue;
    for (float v : g) {
      if (std::fabs(v) > 1e-8f) { any_grad = true; break; }
    }
    if (any_grad) break;
  }
  ASSERT_TRUE(any_grad);
}
