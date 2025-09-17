#pragma once
#include "ag/nn/module.hpp"
#include <cstddef>
#include <vector>

namespace ag::nn {

// BatchNorm2d over NCHW: per-channel mean/var over N*H*W
struct BatchNorm2d : Module {
  std::size_t C;
  float eps;
  float momentum;

  // Learnable scale/shift
  Variable gamma; // [C], requires_grad = true
  Variable beta;  // [C], requires_grad = true

  // Running stats (buffers, no grad)
  Variable running_mean; // [C], requires_grad = false
  Variable running_var;  // [C], requires_grad = false

  BatchNorm2d(std::size_t C, float eps=1e-5, float momentum=0.1);

  Variable forward(const Variable& x) override;

protected:
  // Let the base class collect parameters recursively
  std::vector<Variable*> _parameters() override { return { &gamma, &beta }; }
  void on_mode_change() override {} // optional
};

} // namespace ag::nn
