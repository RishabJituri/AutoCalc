// ============================
// File: include/ag/nn/layers/conv2d.hpp
// ============================
#pragma once

#include <cstddef>
#include <vector>
#include <random>
#include <string>
#include <utility>

#include "ag/nn/module.hpp"
#include "ag/core/variables.hpp"

namespace ag::nn {

// 2D Convolution layer: NCHW layout
// x: [B, C_in, H, W]
// W: [C_out, C_in, KH, KW]
// b: [C_out] (optional)
// y: [B, C_out, H_out, W_out]
struct Conv2d : public Module {
  Conv2d(std::size_t in_channels,
         std::size_t out_channels,
         std::pair<std::size_t,std::size_t> kernel_size,
         std::pair<std::size_t,std::size_t> stride    = {1,1},
         std::pair<std::size_t,std::size_t> padding   = {0,0},
         std::pair<std::size_t,std::size_t> dilation  = {1,1},
         bool bias = true,
         float init_scale = 0.02,
         unsigned long long seed = 0xC0FFEEULL)
  : in_channels_(in_channels), out_channels_(out_channels),
    kernel_(kernel_size), stride_(stride), padding_(padding), dilation_(dilation),
    bias_(bias) {
    init_params_(init_scale, seed);
  }

  Variable forward(const Variable& x) override;

  std::size_t in_channels()  const { return in_channels_; }
  std::size_t out_channels() const { return out_channels_; }
  std::pair<std::size_t,std::size_t> kernel_size() const { return kernel_; }
  std::pair<std::size_t,std::size_t> stride() const { return stride_; }
  std::pair<std::size_t,std::size_t> padding() const { return padding_; }
  std::pair<std::size_t,std::size_t> dilation() const { return dilation_; }
  bool has_bias() const { return bias_; }

protected:
  std::vector<Variable*> _parameters() override {
    if (bias_) return { &W_, &b_ };
    return { &W_ };
  }
  void on_mode_change() override {}

private:
  static Variable make_param_(const std::vector<float>& data,
                              const std::vector<std::size_t>& shape) {
    return Variable(data, shape, /*requires_grad=*/true);
  }
  static std::vector<float> randu_(std::size_t n, float scale, unsigned long long seed) {
    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<float> dist(-scale, scale);
    std::vector<float> v(n);
    for (auto& t : v) t = dist(rng);
    return v;
  }

  void init_params_(float scale, unsigned long long seed);

  std::size_t in_channels_  = 0;
  std::size_t out_channels_ = 0;
  std::pair<std::size_t,std::size_t> kernel_{1,1};
  std::pair<std::size_t,std::size_t> stride_{1,1};
  std::pair<std::size_t,std::size_t> padding_{0,0};
  std::pair<std::size_t,std::size_t> dilation_{1,1};
  bool bias_ = true;

  Variable W_; // [C_out, C_in, KH, KW]
  Variable b_; // [C_out]
};

} // namespace ag::nn
