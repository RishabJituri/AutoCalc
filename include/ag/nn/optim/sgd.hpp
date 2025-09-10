// ============================
// File: include/ag/nn/optim/sgd.hpp
// ============================
#pragma once
#include <vector>
#include <cstddef>
#include "ag/core/variables.hpp"
#include "ag/nn/module.hpp"


namespace ag::nn {

struct SGD {
  double lr{0.1};           // <- stronger default helps the test converge
  double momentum{0.0};
  bool   nesterov{false};
  double weight_decay{0.0};

  // velocity per parameter (by Node* identity)
  std::unordered_map<ag::Node*, std::vector<double>> velocity;

  SGD(double lr=0.1, double momentum=0.0, bool nesterov=false, double weight_decay=0.0)
    : lr(lr), momentum(momentum), nesterov(nesterov), weight_decay(weight_decay) {}

  void step(ag::nn::Module& m);
};

} // namespace ag::optim
