// ============================
// File: include/ag/nn/optim/sgd.hpp
// ============================
#pragma once
#include <vector>
#include <cstddef>
#include "ag/core/variables.hpp"
#include "ag/nn/module.hpp"

namespace ag::optim {

struct SGD {
  explicit SGD(double lr) : lr_(lr) {}
  // Step over a module's parameters
  void step(ag::nn::Module& m) const {
    auto params = m.parameters();
    for (auto* p : params) if (p) step(*p);
  }
  // Step over a list of parameters
  void step(const std::vector<ag::Variable*>& params) const {
    for (auto* p : params) if (p) step(*p);
  }
private:
  void step(ag::Variable& p) const {
    if (!p.n) return;
    auto& v = p.n->value;
    auto& g = p.n->grad;
    if (v.size() != g.size()) return;
    for (std::size_t i = 0; i < v.size(); ++i) v[i] -= lr_ * g[i];
  }
  double lr_ = 1e-2;
};

} // namespace ag::optim
