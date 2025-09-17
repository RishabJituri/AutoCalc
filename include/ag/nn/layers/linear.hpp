// ============================
// File: include/ag/nn/linear.hpp
// ============================
#pragma once

#include <cstddef>
#include <random>
#include <vector>
#include <string>
#include <cassert>

#include "ag/nn/module.hpp"
#include "ag/core/variables.hpp"  // ag::Variable

namespace ag::nn {

// A simple fully-connected layer: y = x @ W + b
// Shapes:
//   x: [B, In]
//   W: [In, Out]
//   b: [Out] (optional; broadcast over batch)
class Linear : public Module {
public:
  Linear(std::size_t in_features, std::size_t out_features, bool bias = true,
         float init_scale = 0.02, unsigned long long seed = 0xC0FFEE)
  : in_features_(in_features), out_features_(out_features), bias_(bias) {
    init_params_(init_scale, seed);
  }

  Variable forward(const Variable& x) override;

  std::size_t in_features()  const { return in_features_; }
  std::size_t out_features() const { return out_features_; }
  bool has_bias() const { return bias_; }

protected:
  std::vector<Variable*> _parameters() override {
    if (bias_) return { &W_, &b_ };
    return { &W_ };
  }

  void on_mode_change() override {}

private:
  // Create a learnable parameter Variable with the given data/shape
  static Variable make_param_(const std::vector<float>& data,
                              const std::vector<std::size_t>& shape) {
    // Variable(vec<float>, vec<size_t>, requires_grad)
    return Variable(data, shape, /*requires_grad=*/true);
  }

  void init_params_(float scale, unsigned long long seed);

  // Helper to fill a vector with U(-scale, scale)
  static std::vector<float> randu_(std::size_t n, float scale, unsigned long long seed) {
    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<float> dist(-scale, scale);
    std::vector<float> v(n);
    for (auto& t : v) t = dist(rng);
    return v;
  }

  std::size_t in_features_  = 0;
  std::size_t out_features_ = 0;
  bool bias_ = true;

  Variable W_; // [In, Out]
  Variable b_; // [Out]
};

} // namespace ag::nn

