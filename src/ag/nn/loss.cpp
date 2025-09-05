#include "ag/nn/loss.hpp"
#include "ag/ops/stats.hpp"
#include "ag/ops/elementwise.hpp"
#include <cmath>
#include <stdexcept>
#include <memory>

namespace ag::nn {

Variable cross_entropy(const Variable& logits, const std::vector<std::size_t>& targets) {
  const auto& shp = logits.n->shape;              // expect [B, C]
  if (shp.size() != 2) throw std::invalid_argument("cross_entropy: expect [B,C] logits");
  const std::size_t B = shp[0], C = shp[1];
  if (targets.size() != B) throw std::invalid_argument("cross_entropy: targets size mismatch");

  // forward: mean over batch of (logsumexp - gold_logit)
  auto LSE = logsumexp(logits, {1}, /*keepdims=*/false);  // [B]

  // gather gold logits
  std::vector<double> gold(B, 0.0);
  for (std::size_t b = 0; b < B; ++b) {
    auto t = targets[b];
    if (t >= C) throw std::invalid_argument("cross_entropy: target out of range");
    gold[b] = logits.n->value[b*C + t];
  }

  // build scalar loss node explicitly
  auto loss = std::make_shared<Node>();
  loss->shape = {};                 // scalar
  loss->value = {0.0};
  loss->grad  = {0.0};
  loss->requires_grad = logits.n->requires_grad;
  loss->parents = {logits.n};

  // compute mean loss value
  double acc = 0.0;
  for (std::size_t b = 0; b < B; ++b) acc += (LSE.n->value[b] - gold[b]);
  loss->value[0] = acc / double(B);

  // backward: dL/dlogits = (softmax - one_hot) / B
  std::weak_ptr<Node> oweak = loss;
  loss->backward = [Xn = logits.n, B, C, targets, oweak]() {
    if (!Xn || !Xn->requires_grad) return;
    auto op = oweak.lock(); if (!op) return;
    Node* o = op.get();
    const double seed = o->grad[0];  // scalar upstream grad

    for (std::size_t b = 0; b < B; ++b) {
      // stable softmax
      double m = -1e300;
      for (std::size_t c = 0; c < C; ++c) m = std::max(m, Xn->value[b*C + c]);
      double Z = 0.0;
      for (std::size_t c = 0; c < C; ++c) Z += std::exp(Xn->value[b*C + c] - m);

      for (std::size_t c = 0; c < C; ++c) {
        double p = std::exp(Xn->value[b*C + c] - m) / Z;
        double g = p - (c == targets[b] ? 1.0 : 0.0);
        Xn->grad[b*C + c] += (g / double(B)) * seed;
      }
    }
  };

  return make_from_node(loss);
}

} // namespace ag::nn
