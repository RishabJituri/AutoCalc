#include "ag/ops/graph.hpp"
#include "ag/ops/tensor_utils.hpp"

namespace ag {

Variable stop_gradient(const Variable& x) {
  auto n = std::make_shared<Node>();
  n->shape = x.n->shape;
  n->value = x.n->value;
  n->grad.assign(n->value.size(), 0.0);
  n->requires_grad = false;
  n->parents.clear();
  n->backward = []() {}; // no-op
  return make_from_node(n);
}

Variable detach(const Variable& x) {
  return stop_gradient(x); // your existing implementation
}
} // namespace ag

