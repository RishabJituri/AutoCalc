#include "ag/ops/graph.hpp"

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
  return stop_gradient(x);
}

NoGradGuard::NoGradGuard()  { ag::set_grad_enabled(false); }
NoGradGuard::~NoGradGuard() { ag::set_grad_enabled(true);  }
} // namespace ag
