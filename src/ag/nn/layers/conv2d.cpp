// ============================
// File: src/ag/nn/layers/conv2d.cpp
// ============================
#include "ag/nn/layers/conv2d.hpp"
#include "ag/ops/tensor_utils.hpp"
#include "ag/parallel/parallel_for.hpp"
#include "ag/ops/gemm.hpp"
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
  // register weight
  register_parameter("weight", W_);

  // b: [C_out]
  if (bias_) {
    b_ = make_param_(randu_(Cout, scale, seed + 1337ULL), {Cout});
    register_parameter("bias", b_);
  } else {
    b_ = Variable(std::vector<float>{}, {0}, /*requires_grad=*/false);
  }
}

// expose local params to Module system
std::vector<ag::Variable*> Conv2d::_parameters() {
  std::vector<ag::Variable*> out;
  out.push_back(&W_);
  if (bias_) out.push_back(&b_);
  return out;
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

  // Forward compute (im2col + matmul)
  const std::size_t K = Cin * KH * KW;
  const std::size_t rows = B * H_out * W_out;

  // pack W into (K, Cout) once
  std::vector<float> Wcol(K * Cout);
  for (std::size_t oc = 0; oc < Cout; ++oc) {
    for (std::size_t ic = 0; ic < Cin; ++ic) {
      for (std::size_t kh = 0; kh < KH; ++kh) {
        for (std::size_t kw = 0; kw < KW; ++kw) {
          const std::size_t kidx = (ic * KH + kh) * KW + kw;
          const std::size_t w_off = oc * wstr[0] + ic * wstr[1] + kh * wstr[2] + kw * wstr[3];
          Wcol[kidx * Cout + oc] = W_.n->value[w_off];
        }
      }
    }
  }

  // process rows in blocks to limit peak im2col memory
#ifdef AG_GEMM_MC
  const std::size_t ROW_BLOCK = std::max<std::size_t>(1, AG_GEMM_MC);
#else
  const std::size_t ROW_BLOCK = 256;
#endif
  const std::size_t num_blocks = (rows + ROW_BLOCK - 1) / ROW_BLOCK;
  const auto xdata = x.n->value.data();

  ag::parallel::parallel_for(num_blocks, /*grain=*/1, [&](std::size_t b0, std::size_t b1){
    for (std::size_t bi = b0; bi < b1; ++bi) {
      const std::size_t r0 = bi * ROW_BLOCK;
      const std::size_t r1 = std::min(rows, r0 + ROW_BLOCK);
      const std::size_t rb = r1 - r0;
      std::vector<float> im2col_block(rb * K);

      for (std::size_t rr = 0; rr < rb; ++rr) {
        const std::size_t r = r0 + rr;
        const std::size_t b = r / (H_out * W_out);
        std::size_t tmp = r % (H_out * W_out);
        const std::size_t oh = tmp / W_out;
        const std::size_t ow = tmp % W_out;
        const int in_h0 = int(oh * SH) - int(PH);
        const int in_w0 = int(ow * SW) - int(PW);
        const std::size_t base_x_b = b * xstr[0];

        std::size_t idx = 0;
        for (std::size_t ic = 0; ic < Cin; ++ic) {
          for (std::size_t kh = 0; kh < KH; ++kh) {
            const int ih = in_h0 + int(kh * DH);
            for (std::size_t kw = 0; kw < KW; ++kw) {
              const int iw = in_w0 + int(kw * DW);
              float v = 0.0f;
              if (ih >= 0 && ih < int(H) && iw >= 0 && iw < int(W)) {
                const std::size_t x_off = base_x_b + ic * xstr[1] + std::size_t(ih) * xstr[2] + std::size_t(iw) * xstr[3];
                v = xdata[x_off];
              }
              im2col_block[rr * K + idx] = v;
              ++idx;
            }
          }
        }
      }

      // compute matmul for this block
      Variable Ablock(im2col_block, {rb, K}, /*requires_grad=*/false);
      Variable Bmat(Wcol, {K, Cout}, /*requires_grad=*/false);
      Variable Yblock = ag::matmul(Ablock, Bmat);
      const auto ydata = Yblock.n->value.data();

      // scatter block results
      for (std::size_t rr = 0; rr < rb; ++rr) {
        const std::size_t r = r0 + rr;
        const std::size_t b = r / (H_out * W_out);
        std::size_t tmp = r % (H_out * W_out);
        const std::size_t oh = tmp / W_out;
        const std::size_t ow = tmp % W_out;
        for (std::size_t oc = 0; oc < Cout; ++oc) {
          const std::size_t y_off = ((b * Cout + oc) * H_out + oh) * W_out + ow;
          out->value[y_off] = ydata[rr * Cout + oc] + (bias_ ? b_.n->value[oc] : 0.0f);
        }
      }
    }
  });

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

    // Dimensions for im2col/GEMM
    const std::size_t K = Cin * KH * KW;
    const std::size_t rows = B * H_out * W_out;
#ifdef AG_GEMM_MC
    const std::size_t ROW_BLOCK = std::max<std::size_t>(1, AG_GEMM_MC);
#else
    const std::size_t ROW_BLOCK = 256;
#endif
    const std::size_t num_blocks = (rows + ROW_BLOCK - 1) / ROW_BLOCK;

    // Allocate per-block partials for dW and dbias
    std::vector<std::vector<float>> partials(num_blocks);
    std::vector<std::vector<float>> bias_partials(num_blocks);
    for (std::size_t bi = 0; bi < num_blocks; ++bi) {
      const std::size_t r0 = bi * ROW_BLOCK;
      const std::size_t r1 = std::min(rows, r0 + ROW_BLOCK);
      const std::size_t rb = r1 - r0;
      partials[bi].assign(K * Cout, 0.0f);
      bias_partials[bi].assign(Cout, 0.0f);
    }

    const auto odata = o->grad.data();
    const auto xdata = (xnode ? xnode->value.data() : nullptr);

    // Parallel compute partial dW and bias sums per block
    ag::parallel::parallel_for(num_blocks, /*grain=*/1, [&](std::size_t b0, std::size_t b1){
      for (std::size_t bi = b0; bi < b1; ++bi) {
        const std::size_t r0 = bi * ROW_BLOCK;
        const std::size_t r1 = std::min(rows, r0 + ROW_BLOCK);
        const std::size_t rb = r1 - r0;
        if (rb == 0) continue;

        // build im2col_block (rb x K) and Ygrad_block (rb x Cout)
        std::vector<float> im2col_block(rb * K);
        std::vector<float> Yg_block(rb * Cout);

        for (std::size_t rr = 0; rr < rb; ++rr) {
          const std::size_t r = r0 + rr;
          const std::size_t b = r / (H_out * W_out);
          std::size_t tmp = r % (H_out * W_out);
          const std::size_t oh = tmp / W_out;
          const std::size_t ow_ = tmp % W_out;
          const int in_h0 = int(oh * SH) - int(PH);
          const int in_w0 = int(ow_ * SW) - int(PW);
          const std::size_t base_x_b = b * xstr[0];

          // fill im2col row
          std::size_t idx = 0;
          for (std::size_t ic = 0; ic < Cin; ++ic) {
            for (std::size_t kh = 0; kh < KH; ++kh) {
              const int ih = in_h0 + int(kh * DH);
              for (std::size_t kw = 0; kw < KW; ++kw) {
                const int iw = in_w0 + int(kw * DW);
                float v = 0.0f;
                if (ih >= 0 && ih < int(H) && iw >= 0 && iw < int(W) && xdata) {
                  const std::size_t x_off = base_x_b + ic * xstr[1] + std::size_t(ih) * xstr[2] + std::size_t(iw) * xstr[3];
                  v = xdata[x_off];
                }
                im2col_block[rr * K + idx] = v;
                ++idx;
              }
            }
          }

          // fill Ygrad row and bias partial
          for (std::size_t oc = 0; oc < Cout; ++oc) {
            const std::size_t y_off = ((b * Cout + oc) * H_out + oh) * W_out + ow_;
            const float go = odata[y_off];
            Yg_block[rr * Cout + oc] = go;
            bias_partials[bi][oc] += go;
          }
        }

        // Transpose im2col to (K x rb) for matmul A^T * Yg
        std::vector<float> Atrans(K * rb);
        for (std::size_t rr = 0; rr < rb; ++rr) {
          for (std::size_t k = 0; k < K; ++k) {
            Atrans[k * rb + rr] = im2col_block[rr * K + k];
          }
        }

        // Compute partial = Atrans (K x rb) * Yg_block (rb x Cout) -> (K x Cout)
        Variable AtransV(Atrans, {K, rb}, /*requires_grad=*/false);
        Variable YgV( Yg_block, {rb, Cout}, /*requires_grad=*/false);
        Variable Part = ag::matmul(AtransV, YgV);

        // copy partial into storage
        const auto partv = Part.n->value.data();
        std::copy(partv, partv + (K * Cout), partials[bi].begin());
      }
    });

    // Deterministic merge of partials into global wnode->grad and bias
    if (wnode && wnode->requires_grad) {
      if (wnode->grad.size() != wnode->value.size()) wnode->grad.assign(wnode->value.size(), 0.0f);
      for (std::size_t bi = 0; bi < num_blocks; ++bi) {
        const auto &P = partials[bi];
        for (std::size_t kidx = 0; kidx < K; ++kidx) {
          const std::size_t ic = kidx / (KH * KW);
          const std::size_t rem = kidx % (KH * KW);
          const std::size_t kh = rem / KW;
          const std::size_t kw = rem % KW;
          for (std::size_t oc = 0; oc < Cout; ++oc) {
            const std::size_t w_off = oc * wstr[0] + ic * wstr[1] + kh * wstr[2] + kw * wstr[3];
            wnode->grad[w_off] += P[kidx * Cout + oc];
          }
        }
      }
    }

    if (bnode && bnode->requires_grad) {
      if (bnode->grad.size() != Cout) bnode->grad.assign(Cout, 0.0f);
      for (std::size_t bi = 0; bi < num_blocks; ++bi) {
        for (std::size_t oc = 0; oc < Cout; ++oc) bnode->grad[oc] += bias_partials[bi][oc];
      }
    }

    // Compute dX using original simple nested loops (keeps correctness)
    if (xnode && xnode->requires_grad) {
      if (xnode->grad.size() != xnode->value.size()) xnode->grad.assign(xnode->value.size(), 0.0f);
      for (std::size_t b = 0; b < B; ++b) {
        for (std::size_t oc = 0; oc < Cout; ++oc) {
          for (std::size_t oh = 0; oh < H_out; ++oh) {
            for (std::size_t ow = 0; ow < W_out; ++ow) {
              const std::size_t y_off = ((b*Cout + oc)*H_out + oh)*W_out + ow;
              const float go = o->grad[y_off];

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
    }
  };

  return make_from_node(out);
}

} // namespace ag::nn
