#include "ag/ops/activations.hpp"
#include "ag/ops/tensor_utils.hpp"
#include <cmath>
#include <memory>   // <-- important for weak_ptr

namespace ag {
using detail::numel;

Variable relu(const Variable& X) {
  auto out = std::make_shared<Node>();
  out->shape = X.n->shape;
  out->value.resize(numel(out->shape));
  out->grad.resize(numel(out->shape), 0.0);
  out->requires_grad = X.n->requires_grad;
  out->parents = {X.n};

  for (std::size_t i = 0; i < out->value.size(); ++i) {
    double v = X.n->value[i];
    out->value[i] = v > 0.0 ? v : 0.0;
  }

  out->backward = [Xn = X.n, oweak = std::weak_ptr<Node>(out)]() {
    auto op = oweak.lock(); if (!op) return;
    Node* o = op.get();
    if (!Xn || !Xn->requires_grad) return;
    for (std::size_t i = 0; i < o->value.size(); ++i) {
      double v = Xn->value[i];
      Xn->grad[i] += (v > 0.0 ? 1.0 : 0.0) * o->grad[i];
    }
  };
  return make_from_node(out);
}

Variable sigmoid(const Variable& X) {
  auto out = std::make_shared<Node>();
  out->shape = X.n->shape;
  out->value.resize(numel(out->shape));
  out->grad.resize(numel(out->shape), 0.0);
  out->requires_grad = X.n->requires_grad;
  out->parents = {X.n};

  for (std::size_t i = 0; i < out->value.size(); ++i) {
    out->value[i] = 1.0 / (1.0 + std::exp(-X.n->value[i]));
  }

  out->backward = [Xn = X.n, oweak = std::weak_ptr<Node>(out)]() {
    auto op = oweak.lock(); if (!op) return;
    Node* o = op.get();
    if (!Xn || !Xn->requires_grad) return;
    for (std::size_t i = 0; i < o->value.size(); ++i) {
      double y = o->value[i];                 // sigmoid(x) cached in forward
      Xn->grad[i] += (y * (1.0 - y)) * o->grad[i];
    }
  };
  return make_from_node(out);
}

Variable tanhv(const Variable& X) {
  auto out = std::make_shared<Node>();
  out->shape = X.n->shape;
  out->value.resize(numel(out->shape));
  out->grad.resize(numel(out->shape), 0.0);
  out->requires_grad = X.n->requires_grad;
  out->parents = {X.n};

  for (std::size_t i = 0; i < out->value.size(); ++i) {
    out->value[i] = std::tanh(X.n->value[i]);
  }

  out->backward = [Xn = X.n, oweak = std::weak_ptr<Node>(out)]() {
    auto op = oweak.lock(); if (!op) return;
    Node* o = op.get();
    if (!Xn || !Xn->requires_grad) return;
    for (std::size_t i = 0; i < o->value.size(); ++i) {
      double y = o->value[i];                 // tanh(x)
      Xn->grad[i] += (1.0 - y * y) * o->grad[i];
    }
  };
  return make_from_node(out);
}

Variable logv(const Variable& X) {
  auto out = std::make_shared<Node>();
  out->shape = X.n->shape;
  out->value.resize(numel(out->shape));
  out->grad.resize(numel(out->shape), 0.0);
  out->requires_grad = X.n->requires_grad;
  out->parents = {X.n};

  for (std::size_t i = 0; i < out->value.size(); ++i) {
    out->value[i] = std::log(X.n->value[i]);  // domain: X>0
  }

  out->backward = [Xn = X.n, oweak = std::weak_ptr<Node>(out)]() {
    auto op = oweak.lock(); if (!op) return;
    Node* o = op.get();
    if (!Xn || !Xn->requires_grad) return;
    for (std::size_t i = 0; i < o->value.size(); ++i) {
      Xn->grad[i] += (1.0 / Xn->value[i]) * o->grad[i];
    }
  };
  return make_from_node(out);
}

Variable clamp(const Variable& X, double lo, double hi) {
  auto out = std::make_shared<Node>();
  out->shape = X.n->shape;
  out->value.resize(numel(out->shape));
  out->grad.resize(numel(out->shape), 0.0);
  out->requires_grad = X.n->requires_grad;
  out->parents = {X.n};

  for (std::size_t i = 0; i < out->value.size(); ++i) {
    double v = X.n->value[i];
    if (v < lo) v = lo;
    if (v > hi) v = hi;
    out->value[i] = v;
  }

  out->backward = [Xn = X.n, lo, hi, oweak = std::weak_ptr<Node>(out)]() {
    auto op = oweak.lock(); if (!op) return;
    Node* o = op.get();
    if (!Xn || !Xn->requires_grad) return;
    for (std::size_t i = 0; i < o->value.size(); ++i) {
      double v = Xn->value[i];
      double g = (v <= lo || v >= hi) ? 0.0 : 1.0;   // subgradient
      Xn->grad[i] += g * o->grad[i];
    }
  };
  return make_from_node(out);
}

} // namespace ag
