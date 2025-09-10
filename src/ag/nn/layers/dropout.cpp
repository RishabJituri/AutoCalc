#include "ag/nn/layers/dropout.hpp"
#include "ag/ops/tensor_utils.hpp"
#include <random>
#include <memory>

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

  std::mt19937_64 rng(seed ? seed : std::random_device{}());   // <- correct helper
  std::bernoulli_distribution bern(1.0 - p);
  const double scale = 1.0 / (1.0 - p);

  std::vector<uint8_t> mask(out->value.size(), 0);

  for (std::size_t i = 0; i < out->value.size(); ++i) {
    const bool keep = bern(rng);
    mask[i] = keep ? 1u : 0u;
    out->value[i] = keep ? X.n->value[i] * scale : 0.0;
  }

  out->backward = [Xn = X.n, mask = std::move(mask), scale,
                   oweak = std::weak_ptr<ag::Node>(out)]() {
    auto op = oweak.lock(); if (!op) return;
    ag::Node* o = op.get();

    if (!Xn || !Xn->requires_grad) return;
    for (std::size_t i = 0; i < o->value.size(); ++i) {
      if (mask[i]) Xn->grad[i] += scale * o->grad[i];
    }
  };

  return make_from_node(out);
}

} // namespace ag::nn
