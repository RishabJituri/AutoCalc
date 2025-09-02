// ============================
// File: src/ag/nn/layers/lstm.cpp
// ============================
#include "ag/nn/layers/lstm.hpp"
#include "ag/ops/elementwise.hpp"
#include "ag/ops/tensor_utils.hpp"

#include <stdexcept>

namespace ag {
  // Small helpers built from existing elementwise ops
  static Variable const_scalar_like(const Variable& ref, double v) {
    const auto& sh = ref.shape();
    std::vector<double> data(ref.value().size(), v);
    return Variable(data, sh, /*requires_grad=*/false);
  }
  static Variable sigmoid(const Variable& x) {
    // 1 / (1 + exp(-x))
    auto one = const_scalar_like(x, 1.0);
    return div(one, add(one, expv(neg(x))));
  }
  static Variable tanh_v(const Variable& x) {
    // tanh(x) = (exp(2x) - 1) / (exp(2x) + 1)
    auto two = const_scalar_like(x, 2.0);
    auto e2x = expv(mul(two, x));
    auto one = const_scalar_like(x, 1.0);
    return div(sub(e2x, one), add(e2x, one));
  }
}

namespace ag::nn {

using ag::detail::strides_for;
using ag::detail::numel;
using ag::sigmoid;
using ag::tanh_v;

void LSTMCell::init_params_(double scale, unsigned long long seed) {
  const std::size_t I = input_size_, H = hidden_size_;
  W_ii_ = make_param_(randu_(I*H, scale, seed + 0), {I, H});
  W_if_ = make_param_(randu_(I*H, scale, seed + 1), {I, H});
  W_ig_ = make_param_(randu_(I*H, scale, seed + 2), {I, H});
  W_io_ = make_param_(randu_(I*H, scale, seed + 3), {I, H});

  W_hi_ = make_param_(randu_(H*H, scale, seed + 4), {H, H});
  W_hf_ = make_param_(randu_(H*H, scale, seed + 5), {H, H});
  W_hg_ = make_param_(randu_(H*H, scale, seed + 6), {H, H});
  W_ho_ = make_param_(randu_(H*H, scale, seed + 7), {H, H});

  if (bias_) {
    b_i_ = make_param_(randu_(H, scale, seed + 10), {H});
    b_f_ = make_param_(randu_(H, scale, seed + 11), {H});
    b_g_ = make_param_(randu_(H, scale, seed + 12), {H});
    b_o_ = make_param_(randu_(H, scale, seed + 13), {H});
  } else {
    b_i_ = Variable(std::vector<double>{}, {0}, /*requires_grad=*/false);
    b_f_ = Variable(std::vector<double>{}, {0}, /*requires_grad=*/false);
    b_g_ = Variable(std::vector<double>{}, {0}, /*requires_grad=*/false);
    b_o_ = Variable(std::vector<double>{}, {0}, /*requires_grad=*/false);
  }
}

std::vector<Variable*> LSTMCell::_parameters() {
  std::vector<Variable*> ps = {
    &W_ii_, &W_if_, &W_ig_, &W_io_,
    &W_hi_, &W_hf_, &W_hg_, &W_ho_
  };
  if (bias_) {
    ps.push_back(&b_i_);
    ps.push_back(&b_f_);
    ps.push_back(&b_g_);
    ps.push_back(&b_o_);
  }
  return ps;
}

std::pair<Variable, Variable> LSTMCell::forward_step(const Variable& x,
                                                     const Variable& h,
                                                     const Variable& c) {
  // Shapes: x:[B,I], h:[B,H], c:[B,H]
  const auto& xs = x.shape();
  const auto& hs = h.shape();
  const auto& cs = c.shape();
  if (xs.size() != 2) throw std::invalid_argument("LSTMCell expects x shape [B,I]");
  if (hs.size() != 2 || cs.size() != 2) throw std::invalid_argument("LSTMCell expects h,c shape [B,H]");
  if (xs[0] != hs[0] || hs[0] != cs[0]) throw std::invalid_argument("LSTMCell batch mismatch");
  if (xs[1] != input_size_) throw std::invalid_argument("LSTMCell input_size mismatch");
  if (hs[1] != hidden_size_ || cs[1] != hidden_size_) throw std::invalid_argument("LSTMCell hidden_size mismatch");

  // Gates
  auto xi = add(matmul(x, W_ii_), matmul(h, W_hi_));
  auto xf = add(matmul(x, W_if_), matmul(h, W_hf_));
  auto xg = add(matmul(x, W_ig_), matmul(h, W_hg_));
  auto xo = add(matmul(x, W_io_), matmul(h, W_ho_));
  if (bias_) {
    xi = add(xi, b_i_);
    xf = add(xf, b_f_);
    xg = add(xg, b_g_);
    xo = add(xo, b_o_);
  }

  auto i = sigmoid(xi);
  auto f = sigmoid(xf);
  auto g = tanh_v(xg);
  auto o = sigmoid(xo);

  auto c_next = add(mul(f, c), mul(i, g));
  auto h_next = mul(o, tanh_v(c_next));
  return {h_next, c_next};
}

Variable LSTMCell::forward(const Variable& x) {
  // Zero initial state convenience (NoGrad)
  const auto& xs = x.shape();
  if (xs.size() != 2) throw std::invalid_argument("LSTMCell::forward expects x shape [B,I]");
  const std::size_t B = xs[0], H = hidden_size_;
  std::vector<double> zh(B*H, 0.0);
  Variable h0(zh, {B,H}, /*requires_grad=*/false);
  Variable c0(zh, {B,H}, /*requires_grad=*/false);
  return forward_step(x, h0, c0).first;
}

} // namespace ag::nn
