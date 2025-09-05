#include "ag/ops/reshape.hpp"
#include "ag/ops/tensor_utils.hpp"
#include <numeric>

namespace ag {
using detail::numel;

Variable flatten(const Variable& X, std::size_t start_dim) {
  const auto& shp = X.n->shape;
  const std::size_t rank = shp.size();
  if (start_dim >= rank) return X; // no-op

  std::size_t prefix = 1;
  for (std::size_t i = 0; i < start_dim; ++i) prefix *= shp[i];
  std::size_t suffix = 1;
  for (std::size_t i = start_dim; i < rank; ++i) suffix *= shp[i];

  auto out = std::make_shared<Node>();
  out->shape = {prefix, suffix};
  out->value = X.n->value;               // same data layout; copy is fine
  out->grad.assign(numel(out->shape), 0.0);
  out->requires_grad = X.n->requires_grad;
  out->parents = {X.n};

  out->backward = [Xn = X.n, oweak = std::weak_ptr<ag::Node>(out)]() {
    auto op = oweak.lock(); if (!op) return;
    ag::Node* o = op.get();

    if (!Xn || !Xn->requires_grad) return;
    // 1-1 mapping of elements
    for (std::size_t i = 0; i < o->grad.size(); ++i) {
      Xn->grad[i] += o->grad[i];
    }
  };
  return make_from_node(out);
}

} // namespace ag
