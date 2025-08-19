#include "variable.hpp"
#include <unordered_set>
#include <cmath>

namespace ag {

static void topo_collect(const std::shared_ptr<Node>& node,
                         std::vector<std::shared_ptr<Node>>& order,
                         std::unordered_set<Node*>& seen) {
  if (!node || seen.count(node.get())) return;
  seen.insert(node.get());
  for (auto& p : node->parents) topo_collect(p, order, seen);
  order.push_back(node);
}

Variable::Variable() : n(std::make_shared<Node>()) {}
Variable::Variable(double v, bool req) : Variable() {
  n->value = v; n->requires_grad = req;
}

double Variable::value() const { return n->value; }
double Variable::grad()  const { return n->grad;  }
bool   Variable::requires_grad() const { return n->requires_grad; }

void Variable::zero_grad() {
  std::vector<std::shared_ptr<Node>> order;
  std::unordered_set<Node*> seen;
  topo_collect(n, order, seen);
  for (auto& x : order) x->grad = 0.0;
}

void Variable::backward(double seed) {
  // Collect subgraph in topological order (parents first)
  std::vector<std::shared_ptr<Node>> order;
  std::unordered_set<Node*> seen;
  topo_collect(n, order, seen);

  // Seed output gradient
  n->grad += seed;

  // Reverse traversal: call node backward closures
  for (auto it = order.rbegin(); it != order.rend(); ++it) {
    auto& node = *it;
    if (node->backward) node->backward();
  }
}

// ===== helpers =====
static inline bool any_req(const std::shared_ptr<Node>& a,
                           const std::shared_ptr<Node>& b = nullptr) {
  return a->requires_grad || (b && b->requires_grad);
}

// ===== ops =====

Variable add(const Variable& a, const Variable& b) {
  auto out = std::make_shared<Node>();
  out->value = a.n->value + b.n->value;
  out->parents = { a.n, b.n };
  out->requires_grad = any_req(a.n, b.n);

  std::weak_ptr<Node> out_w = out, a_w = a.n, b_w = b.n;
  out->backward = [out_w, a_w, b_w]() {
    auto out_n = out_w.lock(); if (!out_n) return;
    if (auto an = a_w.lock(); an && an->requires_grad) an->grad += out_n->grad;
    if (auto bn = b_w.lock(); bn && bn->requires_grad) bn->grad += out_n->grad;
  };
  return Variable(out);
}

Variable sub(const Variable& a, const Variable& b) {
  auto out = std::make_shared<Node>();
  out->value = a.n->value - b.n->value;
  out->parents = { a.n, b.n };
  out->requires_grad = any_req(a.n, b.n);

  std::weak_ptr<Node> out_w = out, a_w = a.n, b_w = b.n;
  out->backward = [out_w, a_w, b_w]() {
    auto out_n = out_w.lock(); if (!out_n) return;
    if (auto an = a_w.lock(); an && an->requires_grad) an->grad +=  out_n->grad;
    if (auto bn = b_w.lock(); bn && bn->requires_grad) bn->grad += -out_n->grad;
  };
  return Variable(out);
}

Variable mul(const Variable& a, const Variable& b) {
  auto out = std::make_shared<Node>();
  out->value = a.n->value * b.n->value;
  out->parents = { a.n, b.n };
  out->requires_grad = any_req(a.n, b.n);

  std::weak_ptr<Node> out_w = out, a_w = a.n, b_w = b.n;
  out->backward = [out_w, a_w, b_w]() {
    auto out_n = out_w.lock(); if (!out_n) return;
    auto an = a_w.lock(); auto bn = b_w.lock();
    if (an && an->requires_grad) an->grad += out_n->grad * (bn ? bn->value : 0.0);
    if (bn && bn->requires_grad) bn->grad += out_n->grad * (an ? an->value : 0.0);
  };
  return Variable(out);
}

Variable div(const Variable& a, const Variable& b) {
  auto out = std::make_shared<Node>();
  out->value = a.n->value / b.n->value;
  out->parents = { a.n, b.n };
  out->requires_grad = any_req(a.n, b.n);

  std::weak_ptr<Node> out_w = out, a_w = a.n, b_w = b.n;
  out->backward = [out_w, a_w, b_w]() {
    auto out_n = out_w.lock(); if (!out_n) return;
    auto an = a_w.lock(); auto bn = b_w.lock();
    const double denom = bn ? bn->value : 1.0;
    if (an && an->requires_grad) an->grad += out_n->grad * (1.0 / denom);
    if (bn && bn->requires_grad) bn->grad += out_n->grad * (- (an ? an->value : 0.0) / (denom*denom));
  };
  return Variable(out);
}

Variable pow(const Variable& a, const Variable& b) {
  auto out = std::make_shared<Node>();
  const double y = a.n->value;
  const double p = b.n->value;
  out->value = std::pow(y, p);
  out->parents = { a.n, b.n };
  out->requires_grad = any_req(a.n, b.n);

  std::weak_ptr<Node> out_w = out, a_w = a.n, b_w = b.n;
  out->backward = [out_w, a_w, b_w]() {
    auto out_n = out_w.lock(); if (!out_n) return;
    auto an = a_w.lock(); auto bn = b_w.lock();
    if (an && an->requires_grad) {
      const double y = an->value;
      const double p = bn ? bn->value : 0.0;
      // d/dy y^p = p * y^(p-1)   (undefined for some y<=0,p combos; you can guard if needed)
      if (!(y == 0.0 && p < 1.0)) an->grad += out_n->grad * (p * std::pow(y, p - 1.0));
    }
    if (bn && bn->requires_grad) {
      const double y = an ? an->value : 1.0;
      const double ln_y = (y > 0.0) ? std::log(y) : 0.0; // guard log for nonpositive y
      bn->grad += out_n->grad * (out_n->value * ln_y);
    }
  };
  return Variable(out);
}

Variable sinv(const Variable& x) {
  auto out = std::make_shared<Node>();
  out->value = std::sin(x.n->value);
  out->parents = { x.n };
  out->requires_grad = x.n->requires_grad;

  std::weak_ptr<Node> out_w = out, x_w = x.n;
  out->backward = [out_w, x_w]() {
    auto out_n = out_w.lock(); if (!out_n) return;
    if (auto xn = x_w.lock(); xn && xn->requires_grad) xn->grad += out_n->grad * std::cos(xn->value);
  };
  return Variable(out);
}

Variable cosv(const Variable& x) {
  auto out = std::make_shared<Node>();
  out->value = std::cos(x.n->value);
  out->parents = { x.n };
  out->requires_grad = x.n->requires_grad;

  std::weak_ptr<Node> out_w = out, x_w = x.n;
  out->backward = [out_w, x_w]() {
    auto out_n = out_w.lock(); if (!out_n) return;
    if (auto xn = x_w.lock(); xn && xn->requires_grad) xn->grad += out_n->grad * (-std::sin(xn->value));
  };
  return Variable(out);
}

Variable expv(const Variable& x) {
  auto out = std::make_shared<Node>();
  out->value = std::exp(x.n->value);
  out->parents = { x.n };
  out->requires_grad = x.n->requires_grad;

  std::weak_ptr<Node> out_w = out, x_w = x.n;
  out->backward = [out_w, x_w]() {
    auto out_n = out_w.lock(); if (!out_n) return;
    if (auto xn = x_w.lock(); xn && xn->requires_grad) xn->grad += out_n->grad * out_n->value;
  };
  return Variable(out);
}

Variable neg(const Variable& x) {
  auto out = std::make_shared<Node>();
  out->value = -x.n->value;
  out->parents = { x.n };
  out->requires_grad = x.n->requires_grad;

  std::weak_ptr<Node> out_w = out, x_w = x.n;
  out->backward = [out_w, x_w]() {
    auto out_n = out_w.lock(); if (!out_n) return;
    if (auto xn = x_w.lock(); xn && xn->requires_grad) xn->grad += -out_n->grad;
  };
  return Variable(out);
}

} // namespace ag
