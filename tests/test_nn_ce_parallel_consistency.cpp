#include "test_framework.hpp"
#include "ag/core/variables.hpp"
#include "ag/nn/loss.hpp"
#include "ag/core/parallel.hpp"
#include <vector>
#include <random>
#include <cmath>

using ag::Variable;
using ag::set_max_threads;

static std::vector<double> randu(std::size_t n, double a=-1.0, double b=1.0) {
  std::mt19937 rng(321);
  std::uniform_real_distribution<double> dist(a,b);
  std::vector<double> v(n);
  for (auto& x : v) x = dist(rng);
  return v;
}

TEST("nn/loss/cross_entropy_parallel_consistency") {
  const std::size_t N = 256;      // rows
  const std::size_t C = 97;       // classes
  Variable logits(randu(N*C), {N, C}, /*requires_grad=*/false);

  // Random integer targets in [0, C-1]
  std::mt19937 rng(42);
  std::uniform_int_distribution<int> uid(0, int(C-1));
  std::vector<std::size_t> targets(N);
  for (std::size_t i = 0; i < N; ++i) targets[i] = std::size_t(uid(rng));

  set_max_threads(1);
  auto L1 = ag::nn::cross_entropy(logits, targets);

  set_max_threads(8);
  auto L2 = ag::nn::cross_entropy(logits, targets);

  double d = std::fabs(L1.value()[0] - L2.value()[0]);
  ASSERT_TRUE(d <= 1e-10);
}
