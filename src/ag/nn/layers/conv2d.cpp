// ============================
// File: src/ag/nn/layers/conv2d.cpp
// ============================
#include "ag/nn/layers/conv2d.hpp"
#include "ag/ops/tensor_utils.hpp"
#include <stdexcept>

namespace ag::nn {
using ag::detail::numel;
using ag::detail::strides_for;
using ag::detail::ravel_index;

static inline std::size_t div_floor(std::size_t a, std::size_t b) { return a / b; }

void Conv2d::init_params_(float scale, unsigned long long seed) {
  const std::size_t Cin = in_channels_;
  const std::size_t Cout = out_channels_;
  const std::size_t KH = kernel_.first, KW = kernel_.second;

  // W: [C_out, C_in, KH, KW]
  const std::size_t nW = Cout * Cin * KH * KW;
  W_ = make_param_(randu_(nW, scale, seed), {Cout, Cin, KH, KW});

  // b: [C_out]
  if (bias_) {
    b_ = make_param_(randu_(Cout, scale, seed + 1337ULL), {Cout});
  } else {
    b_ = Variable(std::vector<float>{}, {0}, /*requires_grad=*/false);
  }
}

Variable Conv2d::forward(const Variable& x) {
  // x: [B, Cin, H, W]
  const auto& xs = x.n->shape;
  if (xs.size() != 4) throw std::invalid_argument("Conv2d expects input [B,C,H,W]");
  const std::size_t B = xs[0], Cin = xs[1], H = xs[2], W = xs[3];
  if (Cin != in_channels_) throw std::invalid_argument("Conv2d: in_channels mismatch");

  const std::size_t Cout = out_channels_;
  const std::size_t KH = kernel_.first, KW = kernel_.second;
  const std::size_t SH = stride_.first, SW = stride_.second;
  const std::size_t PH = padding_.first, PW = padding_.second;
  const std::size_t DH = dilation_.first, DW = dilation_.second;

  const std::size_t H_out = 1 + ( (H + 2*PH) - (DH * (KH - 1) + 1) ) / SH;
  const std::size_t W_out = 1 + ( (W + 2*PW) - (DW * (KW - 1) + 1) ) / SW;
  if ((H + 2*PH) < (DH * (KH - 1) + 1) || (W + 2*PW) < (DW * (KW - 1) + 1))
    throw std::invalid_argument("Conv2d: kernel larger than padded input");

  // Shapes and strides
  const std::vector<std::size_t> wshape = {Cout, Cin, KH, KW};
  const auto wstr = strides_for(wshape);
  const auto xstr = strides_for(xs);
  const std::vector<std::size_t> yshape = {B, Cout, H_out, W_out};

  auto out = std::make_shared<ag::Node>();
  out->shape = yshape;
  out->value.assign(numel(yshape), 0.0);
  out->grad.assign(numel(yshape), 0.0);
  out->requires_grad = (x.n->requires_grad || W_.n->requires_grad || (bias_ && b_.n->requires_grad));
  out->parents = {x.n, W_.n};
  if (bias_) out->parents.push_back(b_.n);

  // Forward compute
  for (std::size_t b = 0; b < B; ++b) {
    for (std::size_t oc = 0; oc < Cout; ++oc) {
      for (std::size_t oh = 0; oh < H_out; ++oh) {
        for (std::size_t ow = 0; ow < W_out; ++ow) {
          float acc = bias_ ? b_.n->value[oc] : 0.0;
          const int in_h0 = int(oh*SH) - int(PH);
          const int in_w0 = int(ow*SW) - int(PW);
          for (std::size_t ic = 0; ic < Cin; ++ic) {
            for (std::size_t kh = 0; kh < KH; ++kh) {
              const int ih = in_h0 + int(kh*DH);
              if (ih < 0 || ih >= int(H)) continue;
              for (std::size_t kw = 0; kw < KW; ++kw) {
                const int iw = in_w0 + int(kw*DW);
                if (iw < 0 || iw >= int(W)) continue;
                const std::size_t x_off = b*xstr[0] + ic*xstr[1] + std::size_t(ih)*xstr[2] + std::size_t(iw)*xstr[3];
                const std::size_t w_off = oc*wstr[0] + ic*wstr[1] + kh*wstr[2] + kw*wstr[3];
                acc += x.n->value[x_off] * W_.n->value[w_off];
              }
            }
          }
          const std::size_t y_off = ((b*Cout + oc)*H_out + oh)*W_out + ow;
          out->value[y_off] = acc;
        }
      }
    }
  }

  // Backward
  std::weak_ptr<ag::Node> ow = out;
  std::weak_ptr<ag::Node> xw = x.n;
  std::weak_ptr<ag::Node> ww = W_.n;
  std::weak_ptr<ag::Node> bw = bias_ ? b_.n : std::weak_ptr<ag::Node>{};

  out->backward = [ow, xw, ww, bw, xs, wshape, yshape, SH, SW, PH, PW, DH, DW]() {
    auto o = ow.lock(); if (!o) return;
    auto xnode = xw.lock();
    auto wnode = ww.lock();
    auto bnode = bw.lock();
    const std::size_t B = xs[0], Cin = xs[1], H = xs[2], W = xs[3];
    const std::size_t Cout = wshape[0], KH = wshape[2], KW = wshape[3];
    const std::size_t H_out = yshape[2], W_out = yshape[3];

    const auto xstr = strides_for(xs);
    const auto wstr = strides_for(wshape);

    for (std::size_t b = 0; b < B; ++b) {
      for (std::size_t oc = 0; oc < Cout; ++oc) {
        for (std::size_t oh = 0; oh < H_out; ++oh) {
          for (std::size_t ow = 0; ow < W_out; ++ow) {
            const std::size_t y_off = ((b*Cout + oc)*H_out + oh)*W_out + ow;
            const float go = o->grad[y_off];

            // bias grad
            if (bnode && bnode->requires_grad) {
              bnode->grad[oc] += go;
            }

            const int in_h0 = int(oh*SH) - int(PH);
            const int in_w0 = int(ow*SW) - int(PW);

            for (std::size_t ic = 0; ic < Cin; ++ic) {
              for (std::size_t kh = 0; kh < KH; ++kh) {
                const int ih = in_h0 + int(kh*DH);
                if (ih < 0 || ih >= int(H)) continue;
                for (std::size_t kw = 0; kw < KW; ++kw) {
                  const int iw = in_w0 + int(kw*DW);
                  if (iw < 0 || iw >= int(W)) continue;

                  const std::size_t x_off = b*xstr[0] + ic*xstr[1] + std::size_t(ih)*xstr[2] + std::size_t(iw)*xstr[3];
                  const std::size_t w_off = oc*wstr[0] + ic*wstr[1] + kh*wstr[2] + kw*wstr[3];

                  // dW += x * go
                  if (wnode && wnode->requires_grad) {
                    wnode->grad[w_off] += (xnode ? xnode->value[x_off] : 0.0) * go;
                  }
                  // dX += W * go
                  if (xnode && xnode->requires_grad) {
                    xnode->grad[x_off] += (wnode ? wnode->value[w_off] : 0.0) * go;
                  }
                }
              }
            }
          }
        }
      }
    }
  };

  return make_from_node(out);
}

} // namespace ag::nn
