#pragma once
#include "ag/nn/module.hpp"
#include <cstdint>
#include <random>  // already there

namespace ag::nn {

struct Dropout : Module {
  float   p{0.5};
  uint64_t seed{0};

  explicit Dropout(float p = 0.5, uint64_t seed = 0) : p(p), seed(seed) {}

  ag::Variable forward(const ag::Variable& x) override;

protected:
  // Dropout has no trainable parameters
  std::vector<ag::Variable*> _parameters() override { return {}; }
  // If your Module declares this as const, use:
  // std::vector<ag::Variable*> _parameters() const override { return {}; }
};

} // namespace ag::nn
