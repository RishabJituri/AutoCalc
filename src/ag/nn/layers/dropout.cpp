#include "ag/nn/layers/dropout.hpp"
#include "ag/ops/tensor_utils.hpp"
#include "ag/parallel/parallel_for.hpp"
#include <random>
#include <memory>
#include <cstdint>

namespace ag::nn {
using ag::Variable;
using ag::Node;
using ag::detail::numel;

Variable Dropout::forward(const Variable& X) {
  if (!training()) return X;                     // <-- use Module::training()

  if (p < 0.0 || p >= 1.0) throw std::invalid_argument("Dropout p must be in [0,1).");

  auto out = std::make_shared<Node>();
  out->shape = X.n->shape;
  out->value.resize(numel(out->shape));
  out->grad.assign(numel(out->shape), 0.0);
  out->requires_grad = X.n->requires_grad;
  out->parents = {X.n};

  // Deterministic counter-based RNG (SplitMix-like) for per-index draws
  auto splitmix32 = [](uint64_t x)->uint32_t {
    x += 0x9E3779B97F4A7C15ull;
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ull;
    x = (x ^ (x >> 27)) * 0x94D049BB133111EBull;
    return uint32_t(x ^ (x >> 31));
  };
  auto bernoulli_bit = [&](uint64_t seed, std::size_t i, float keep)->uint8_t {
    // produce a 24-bit mantissa uniform in [0,1)
    uint32_t r = splitmix32(seed + uint64_t(i) + 1ull);
    float u = float(r >> 8) * (1.0f / 16777216.0f);
    return (u < keep) ? 1u : 0u;
  };

  const std::size_t N = out->value.size();
  const float keepf = 1.0f - static_cast<float>(p);
  const float scalef = 1.0f / std::max(keepf, 1e-8f);
  std::vector<uint8_t> mask(N, 0);

  const std::size_t CUTOFF_BYTES = 128 * 1024; // 128KB
  const std::size_t bytes = N * sizeof(float);
  const auto xin = X.n->value.data();
  auto outp = out->value.data();

  if (bytes < CUTOFF_BYTES) {
    for (std::size_t i = 0; i < N; ++i) {
      uint8_t m = bernoulli_bit(seed, i, keepf);
      mask[i] = m;
      outp[i] = m ? xin[i] * scalef : 0.0f;
    }
  } else {
    const std::size_t GRAIN = 1024;
    ag::parallel::parallel_for(N, GRAIN, [&](std::size_t i0, std::size_t i1){
      for (std::size_t i = i0; i < i1; ++i) {
        uint8_t m = bernoulli_bit(seed, i, keepf);
        mask[i] = m;
        outp[i] = m ? xin[i] * scalef : 0.0f;
      }
    });
  }

  out->backward = [Xn = X.n, mask = std::move(mask), scalef,
                   oweak = std::weak_ptr<ag::Node>(out)]() {
    auto op = oweak.lock(); if (!op) return;
    ag::Node* o = op.get();

    if (!Xn || !Xn->requires_grad) return;
    if (Xn->grad.size() != Xn->value.size()) Xn->grad.assign(Xn->value.size(), 0.0f);

    const std::size_t Nloc = mask.size();
    const std::size_t bytes_loc = Nloc * sizeof(float);
    const std::size_t CUTOFF_BYTES_LOC = 128 * 1024;
    if (bytes_loc < CUTOFF_BYTES_LOC) {
      for (std::size_t i = 0; i < Nloc; ++i) if (mask[i]) Xn->grad[i] += scalef * o->grad[i];
    } else {
      const std::size_t GRAIN = 1024;
      const float* gin = o->grad.data();
      float* gx = Xn->grad.data();
      const uint8_t* mptr = mask.data();
      ag::parallel::parallel_for(Nloc, GRAIN, [&](std::size_t i0, std::size_t i1){
        for (std::size_t i = i0; i < i1; ++i) if (mptr[i]) gx[i] += scalef * gin[i];
      });
    }
  };

  return make_from_node(out);
}

} // namespace ag::nn
