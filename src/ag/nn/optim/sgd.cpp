#include "ag/nn/optim/sgd.hpp"
#include <algorithm>

namespace ag::nn {

void SGD::step(ag::nn::Module& m) {
  auto params = m.parameters();                 // std::vector<ag::Variable*>
  for (ag::Variable* p : params) {
    ag::Node* key = p->n.get();
    auto& v = velocity[key];
    if (v.size() != p->n->value.size()) v.assign(p->n->value.size(), 0.0);

    for (std::size_t i = 0; i < p->n->value.size(); ++i) {
      double g = p->n->grad[i];
      if (weight_decay != 0.0) g += weight_decay * p->n->value[i];  // L2
      v[i] = momentum * v[i] + g;
      const double stepdir = nesterov ? (momentum * v[i] + g) : v[i];
      p->n->value[i] -= lr * stepdir;
    }
    std::fill(p->n->grad.begin(), p->n->grad.end(), 0.0);
  }
}

} // namespace ag::optim
