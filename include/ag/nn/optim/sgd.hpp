// ============================
// File: include/ag/nn/optim/sgd.hpp
// ============================
#pragma once
#include <vector>
#include <cstddef>
#include "ag/core/variables.hpp"
#include "ag/nn/module.hpp"
#include <unordered_map>


namespace ag::nn {

struct SGD {
  float lr{0.1f};           // <- stronger default helps the test converge
  float momentum{0.0f};
  bool   nesterov{false};
  float weight_decay{0.0f};

  // velocity per parameter (by Node* identity)
  std::unordered_map<ag::Node*, std::vector<float>> velocity;

  SGD(float lr=0.1f, float momentum=0.0f, bool nesterov=false, float weight_decay=0.0f)
    : lr(lr), momentum(momentum), nesterov(nesterov), weight_decay(weight_decay) {}

  void step(ag::nn::Module& m);
};

} // namespace ag::optim
