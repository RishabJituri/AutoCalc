// ============================
// File: include/ag/nn/layers/lstm.hpp
// ============================
#pragma once

#include <cstddef>
#include <utility>
#include <vector>
#include <random>

#include "ag/nn/module.hpp"
#include "ag/core/variables.hpp"

namespace ag::nn {

// A single-step LSTM cell (no peepholes), NCHW not relevant (we use [B, I] and [B, H] shapes).
// Equations:
//   i = sigmoid(x W_ii + h W_hi + b_i)
//   f = sigmoid(x W_if + h W_hf + b_f)
//   g = tanh   (x W_ig + h W_hg + b_g)
//   o = sigmoid(x W_io + h W_ho + b_o)
//   c' = f * c + i * g
//   h' = o * tanh(c')
struct LSTMCell : public Module {
  LSTMCell(std::size_t input_size,
           std::size_t hidden_size,
           bool bias = true,
           double init_scale = 0.02,
           unsigned long long seed = 0xABCD1234ULL)
  : input_size_(input_size), hidden_size_(hidden_size), bias_(bias) {
    init_params_(init_scale, seed);
  }

  // Step forward with explicit state. Returns {h_next, c_next}.
  std::pair<Variable, Variable> forward_step(const Variable& x,
                                             const Variable& h,
                                             const Variable& c);

  // Convenience override: assumes zero initial state (NOT recommended for training loops).
  // Returns h_next; you can get c_next by calling forward_step directly.
  Variable forward(const Variable& x) override;

  std::size_t input_size() const { return input_size_; }
  std::size_t hidden_size() const { return hidden_size_; }
  bool has_bias() const { return bias_; }

protected:
  std::vector<Variable*> _parameters() override;

private:
  static Variable make_param_(const std::vector<double>& data,
                              const std::vector<std::size_t>& shape) {
    return Variable(data, shape, /*requires_grad=*/true);
  }
  static std::vector<double> randu_(std::size_t n, double scale, unsigned long long seed) {
    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<double> dist(-scale, scale);
    std::vector<double> v(n);
    for (auto& t : v) t = dist(rng);
    return v;
  }

  void init_params_(double scale, unsigned long long seed);

  std::size_t input_size_ = 0;
  std::size_t hidden_size_ = 0;
  bool bias_ = true;

  // Parameters: 8 weight matrices and 4 bias vectors (if bias_=true)
  // Input weights: [I, H] each
  Variable W_ii_, W_if_, W_ig_, W_io_;
  // Recurrent weights: [H, H] each
  Variable W_hi_, W_hf_, W_hg_, W_ho_;
  // Biases: [H] each
  Variable b_i_, b_f_, b_g_, b_o_;
};

// lstm.hpp (appended)
struct LSTM : Module {
  LSTM(std::size_t input_size, std::size_t hidden_size,
       int num_layers = 1, bool bias = true);

  // X: [B,T,I] -> Y: [B,T,H]
  Variable forward(const Variable& X) override;

  std::vector<Variable*> _parameters() override;

private:
  std::size_t input_size_ = 0, hidden_size_ = 0;
  int  num_layers_ = 1;
  bool bias_ = true;

  // one LSTMCell per layer; weights shared across T steps
  std::vector<std::shared_ptr<LSTMCell>> layers_;
};

} // namespace ag::nn
