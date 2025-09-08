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


// --- appended: general reshape ---
#include <stdexcept>

namespace ag {

Variable reshape(const Variable& X, const std::vector<std::size_t>& new_shape) {
  const auto old_elems = detail::numel(X.n->shape);
  const auto new_elems = detail::numel(new_shape);
  if (old_elems != new_elems) {
    throw std::runtime_error("reshape: number of elements must remain constant");
  }
  if (new_shape == X.n->shape) {  // no-op view
    return X;
  }

  auto out = std::make_shared<Node>();
  out->shape = new_shape;
  out->value = X.n->value;                // same data layout (row-major), copy ok
  out->grad.assign(new_elems, 0.0);
  out->requires_grad = X.n->requires_grad;
  out->parents = {X.n};

  out->backward = [Xn = X.n, oweak = std::weak_ptr<ag::Node>(out)]() {
    auto op = oweak.lock(); if (!op) return;
    ag::Node* o = op.get();
    if (!Xn || !Xn->requires_grad) return;
    // 1-1 element mapping
    for (std::size_t i = 0; i < o->grad.size(); ++i) {
      Xn->grad[i] += o->grad[i];
    }
  };
  return make_from_node(out);
}

// Convenience overload for initializer_list
Variable reshape(const Variable& X, std::initializer_list<std::size_t> new_shape) {
  return reshape(X, std::vector<std::size_t>(new_shape));
}

} // namespace ag
