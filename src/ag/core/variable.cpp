#include "ag/core/variables.hpp"
#include "ag/ops/tensor_utils.hpp"
#include <unordered_set>
#include <stdexcept>

namespace {

// parents-first DFS; reverse(order) used for backprop
void topo_collect(const std::shared_ptr<ag::Node>& node,
                  std::vector<std::shared_ptr<ag::Node>>& order,
                  std::unordered_set<ag::Node*>& seen) {
  if (!node || seen.count(node.get())) return;
  seen.insert(node.get());
  for (auto& p : node->parents) topo_collect(p, order, seen);
  order.push_back(node);
}

} // anon

namespace ag {
using detail::numel;

Variable::Variable() : n(std::make_shared<Node>()) {}
Variable::Variable(std::shared_ptr<Node> node) : n(std::move(node)) {}

Variable::Variable(const std::vector<float>& value,
                   const std::vector<std::size_t>& shape,
                   bool requires_grad) {
  if (numel(shape) != value.size()) throw std::invalid_argument("value size != numel(shape)");
  n = std::make_shared<Node>();
  n->value = value;
  n->shape = shape;
  n->requires_grad = (requires_grad && ag::is_grad_enabled());
  n->grad.assign(value.size(), 0.0);
}

const std::vector<float>& Variable::value() const { return n->value; }
const std::vector<float>& Variable::grad()  const { return n->grad;  }
const std::vector<std::size_t>& Variable::shape() const { return n->shape; }
bool Variable::requires_grad() const { return n->requires_grad; }

Variable make_from_node(std::shared_ptr<Node> node) { return Variable(std::move(node)); }

void Variable::zero_grad() {
  std::vector<std::shared_ptr<Node>> order;
  std::unordered_set<Node*> seen;
  topo_collect(n, order, seen);
  for (auto& x : order) x->grad.assign(numel(x->shape), 0.0);
}

void Variable::backward() {
  if (numel(n->shape) != 1)
    throw std::invalid_argument("Non-scalar output: pass seed to backward(seed)");
  backward(std::vector<float>{1.0});
}

void Variable::backward(const std::vector<float>& seed) {
  std::vector<std::shared_ptr<Node>> order;
  std::unordered_set<Node*> seen;
  topo_collect(n, order, seen);

  if (seed.size() != n->grad.size()) throw std::invalid_argument("Seed grad size mismatch");
  for (std::size_t i = 0; i < seed.size(); ++i) n->grad[i] += seed[i];

  for (auto it = order.rbegin(); it != order.rend(); ++it) {
    auto& node = *it;
    if (node->backward) node->backward();
  }

  // Post-backward cleanup: release the computation graph so intermediate
  // nodes can be freed.  Leaf nodes (no parents) keep their grad intact.
  // Post-backward cleanup: release the computation graph so intermediate
  // nodes can be freed.  Leaf nodes (no parents) keep their grad intact.
  for (auto& node : order) {
    if (node->parents.empty()) continue;   // leaf — keep alive
    node->backward = nullptr;
    node->parents.clear();
  }
}



} // namespace ag
