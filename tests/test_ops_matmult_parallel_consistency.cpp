#include "test_framework.hpp"
#include "ag/core/variables.hpp"
#include "ag/ops/linalg.hpp"
#include "ag/core/parallel.hpp"
#include <vector>
#include <cmath>
#include <random>

using ag::Variable;
using ag::parallel_for;
using ag::set_max_threads;

static std::vector<double> randu(std::size_t n, double a=-1.0, double b=1.0) {
  std::mt19937 rng(123);
  std::uniform_real_distribution<double> dist(a,b);
  std::vector<double> v(n);
  for (auto& x : v) x = dist(rng);
  return v;
}

TEST("ops/matmul/parallel_consistency") {
  const std::size_t M = 37, K = 53, N = 29;
  Variable A(randu(M*K), {M, K}, false);
  Variable B(randu(K*N), {K, N}, false);

  // Single-thread result
  set_max_threads(1);
  auto C1 = ag::matmul(A, B);

  // Multi-thread result
  set_max_threads(8);
  auto C2 = ag::matmul(A, B);

  const auto& v1 = C1.value();
  const auto& v2 = C2.value();
  ASSERT_TRUE(v1.size() == v2.size());
  double max_abs = 0.0;
  for (std::size_t i = 0; i < v1.size(); ++i) {
    double d = std::fabs(v1[i] - v2[i]);
    if (d > max_abs) max_abs = d;
  }
  // Identical in pure double; allow tiny slack
  ASSERT_TRUE(max_abs <= 1e-10);
}
