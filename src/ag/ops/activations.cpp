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
  out->grad.resize(numel(out->shape), 0.0f);
  out->requires_grad = X.n->requires_grad;
  out->parents = {X.n};

  for (std::size_t i = 0; i < out->value.size(); ++i) {
    float v = static_cast<float>(X.n->value[i]);
    out->value[i] = v > 0.0f ? v : 0.0f;
  }

  out->backward = [Xn = X.n, oweak = std::weak_ptr<Node>(out)]() {
    auto op = oweak.lock(); if (!op) return;
    Node* o = op.get();
    if (!Xn || !Xn->requires_grad) return;
    for (std::size_t i = 0; i < o->value.size(); ++i) {
      float v = static_cast<float>(Xn->value[i]);
      Xn->grad[i] += (v > 0.0f ? 1.0f : 0.0f) * static_cast<float>(o->grad[i]);
    }
  };
  return make_from_node(out);
}

Variable sigmoid(const Variable& X) {
  auto out = std::make_shared<Node>();
  out->shape = X.n->shape;
  out->value.resize(numel(out->shape));
  out->grad.resize(numel(out->shape), 0.0f);
  out->requires_grad = X.n->requires_grad;
  out->parents = {X.n};

  for (std::size_t i = 0; i < out->value.size(); ++i) {
    out->value[i] = 1.0f / (1.0f + std::exp(-static_cast<float>(X.n->value[i])));
  }

  out->backward = [Xn = X.n, oweak = std::weak_ptr<Node>(out)]() {
    auto op = oweak.lock(); if (!op) return;
    Node* o = op.get();
    if (!Xn || !Xn->requires_grad) return;
    for (std::size_t i = 0; i < o->value.size(); ++i) {
      float y = static_cast<float>(o->value[i]); // sigmoid(x) cached in forward
      Xn->grad[i] += (y * (1.0f - y)) * static_cast<float>(o->grad[i]);
    }
  };
  return make_from_node(out);
}

Variable tanhv(const Variable& X) {
  auto out = std::make_shared<Node>();
  out->shape = X.n->shape;
  out->value.resize(numel(out->shape));
  out->grad.resize(numel(out->shape), 0.0f);
  out->requires_grad = X.n->requires_grad;
  out->parents = {X.n};

  for (std::size_t i = 0; i < out->value.size(); ++i) {
    out->value[i] = std::tanh(static_cast<float>(X.n->value[i]));
  }

  out->backward = [Xn = X.n, oweak = std::weak_ptr<Node>(out)]() {
    auto op = oweak.lock(); if (!op) return;
    Node* o = op.get();
    if (!Xn || !Xn->requires_grad) return;
    for (std::size_t i = 0; i < o->value.size(); ++i) {
      float y = static_cast<float>(o->value[i]); // tanh(x)
      Xn->grad[i] += (1.0f - y * y) * static_cast<float>(o->grad[i]);
    }
  };
  return make_from_node(out);
}

Variable logv(const Variable& X) {
  auto out = std::make_shared<Node>();
  out->shape = X.n->shape;
  out->value.resize(numel(out->shape));
  out->grad.resize(numel(out->shape), 0.0f);
  out->requires_grad = X.n->requires_grad;
  out->parents = {X.n};

  for (std::size_t i = 0; i < out->value.size(); ++i) {
    out->value[i] = std::log(static_cast<float>(X.n->value[i]));  // domain: X>0
  }

  out->backward = [Xn = X.n, oweak = std::weak_ptr<Node>(out)]() {
    auto op = oweak.lock(); if (!op) return;
    Node* o = op.get();
    if (!Xn || !Xn->requires_grad) return;
    for (std::size_t i = 0; i < o->value.size(); ++i) {
      Xn->grad[i] += (1.0f / static_cast<float>(Xn->value[i])) * static_cast<float>(o->grad[i]);
    }
  };
  return make_from_node(out);
}

Variable clamp(const Variable& X, float lo, float hi) {
  auto out = std::make_shared<Node>();
  out->shape = X.n->shape;
  out->value.resize(numel(out->shape));
  out->grad.resize(numel(out->shape), 0.0f);
  out->requires_grad = X.n->requires_grad;
  out->parents = {X.n};

  for (std::size_t i = 0; i < out->value.size(); ++i) {
    float v = static_cast<float>(X.n->value[i]);
    if (v < lo) v = lo;
    if (v > hi) v = hi;
    out->value[i] = v;
  }

  out->backward = [Xn = X.n, lo, hi, oweak = std::weak_ptr<Node>(out)]() {
    auto op = oweak.lock(); if (!op) return;
    Node* o = op.get();
    if (!Xn || !Xn->requires_grad) return;
    for (std::size_t i = 0; i < o->value.size(); ++i) {
      float v = static_cast<float>(Xn->value[i]);
      float g = (v <= lo || v >= hi) ? 0.0f : 1.0f;   // subgradient
      Xn->grad[i] += g * static_cast<float>(o->grad[i]);
    }
  };
  return make_from_node(out);
}

} // namespace ag
