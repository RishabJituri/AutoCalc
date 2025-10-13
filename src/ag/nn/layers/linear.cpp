// ============================
// File: src/ag/nn/linear.cpp
// ============================
#include "ag/nn/layers/linear.hpp"

// We assume these free functions exist in namespace ag (from your core):
//   Variable matmul(const Variable&, const Variable&);
//   Variable add(const Variable&, const Variable&);  // with row-wise broadcast for bias
namespace ag::nn {

void Linear::init_params_(float scale, unsigned long long seed) {
  const std::size_t in  = in_features_;
  const std::size_t out = out_features_;

  // W shape [In, Out]
  const std::size_t nW = in * out;
  auto wdata = randu_(nW, scale, seed);
  W_ = make_param_(wdata, {in, out});
  register_parameter("weight", W_);

  if (bias_) {
    auto bdata = randu_(out, scale, seed + 1);
    b_ = make_param_(bdata, {out});
    register_parameter("bias", b_);
  }
}

Variable Linear::forward(const Variable& x) {
  // x: [B, In], W: [In, Out] -> y: [B, Out]
  auto y = ag::matmul(x, W_);
  if (bias_) {
    // Expecting add to support row-wise broadcast of [Out] over [B,Out]
    y = ag::add(y, b_);
  }
  return y;
}

std::vector<ag::Variable*> Linear::_parameters() {
  std::vector<ag::Variable*> out;
  out.push_back(&W_);
  if (bias_) out.push_back(&b_);
  return out;
}

} // namespace ag::nn


