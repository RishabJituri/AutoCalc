#pragma once
#include "ag/nn/module.hpp"
#include <cstdint>
#include <random>

namespace ag::nn {

struct Dropout : Module {
  float   p{0.5};
  uint64_t seed{0};
  uint64_t call_counter_{0};   // incremented each forward to vary the mask

  explicit Dropout(float p = 0.5, uint64_t seed = 0)
    : p(p),
      seed(seed == 0 ? std::random_device{}() : seed) {}

  ag::Variable forward(const ag::Variable& x) override;

protected:
  // Dropout has no trainable parameters
  std::vector<ag::Variable*> _parameters() override { return {}; }
};

} // namespace ag::nn
