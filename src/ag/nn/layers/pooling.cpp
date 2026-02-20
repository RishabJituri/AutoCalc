#include "ag/nn/layers/pooling.hpp"
#include "ag/ops/tensor_utils.hpp"
#include "ag/parallel/parallel_for.hpp"
#include <stdexcept>
#include <cmath>
#include <limits>

namespace ag::nn {
using ag::Variable;
using ag::Node;
using ag::detail::numel;
using ag::detail::strides_for;
using ag::detail::ravel_index;

static inline std::size_t safe_out_dim(std::size_t in, std::size_t k, std::size_t s, std::size_t p) {
  if (s == 0) throw std::invalid_argument("Pooling stride must be > 0");
  long long num = (long long)in + 2LL*(long long)p - (long long)k;
  if (num < 0) return 0;
  return 1 + (std::size_t)(num / (long long)s);
}

Variable MaxPool2d::forward(const Variable& X) {
  const auto& xs = X.n->shape; // [B,C,H,W]
  if (xs.size()!=4) throw std::invalid_argument("MaxPool2d expects NCHW");
  std::size_t B=xs[0], C=xs[1], H=xs[2], W=xs[3];
  std::size_t H_out = safe_out_dim(H, kH, sH, pH);
  std::size_t W_out = safe_out_dim(W, kW, sW, pW);
  std::vector<std::size_t> yshape = {B,C,H_out,W_out};

  auto out = std::make_shared<Node>();
  out->shape = yshape;
  out->value.assign(numel(yshape), -std::numeric_limits<float>::infinity());
  if (X.n->requires_grad) out->grad.assign(numel(yshape), 0.0f);
  out->requires_grad = X.n->requires_grad;
  out->parents = {X.n};

  const auto xstr = strides_for(xs);
  const auto ystr = strides_for(yshape);

  // forward: parallel over output elements
  const std::size_t outN = numel(yshape);
  const std::size_t OUT_GRAIN = 1024;
  ag::parallel::parallel_for(outN, OUT_GRAIN, [&](std::size_t i0, std::size_t i1){
    const std::size_t strideB = C * H_out * W_out;
    const std::size_t strideC = H_out * W_out;
    const std::size_t strideH = W_out;
    for (std::size_t lin = i0; lin < i1; ++lin) {
      std::size_t tmp = lin;
      const std::size_t b = tmp / strideB; tmp %= strideB;
      const std::size_t c = tmp / strideC; tmp %= strideC;
      const std::size_t oh = tmp / strideH;
      const std::size_t ow = tmp % strideH;

      const std::size_t base = b * xstr[0] + c * xstr[1];
      const std::size_t h0 = (oh * sH > pH) ? (oh * sH - pH) : 0;
      const std::size_t h1 = std::min(h0 + kH, H);
      const std::size_t w0 = (ow * sW > pW) ? (ow * sW - pW) : 0;
      const std::size_t w1 = std::min(w0 + kW, W);

      float m = -std::numeric_limits<float>::infinity();
      for (std::size_t ih = h0; ih < h1; ++ih) {
        const std::size_t row = base + ih * xstr[2];
        for (std::size_t iw = w0; iw < w1; ++iw) {
          const std::size_t xi = row + iw * xstr[3];
          m = std::max(m, X.n->value[xi]);
        }
      }
      out->value[lin] = m;
    }
  });

  // backward: parallelize over (B,C) channels to avoid races on Xn->grad
  out->backward = [kH_=this->kH, kW_=this->kW, sH_=this->sH, sW_=this->sW,
                   pH_=this->pH, pW_=this->pW,
                   Xn = X.n, xs, yshape,
                   oweak = std::weak_ptr<ag::Node>(out)]() {
    auto op = oweak.lock(); if (!op) return;
    ag::Node* o = op.get();

    if (!Xn || !Xn->requires_grad) return;
    if (Xn->grad.size() != Xn->value.size()) Xn->grad.assign(Xn->value.size(), 0.0f);

    const auto xstr = strides_for(xs);
    const auto ystr = strides_for(yshape);
    const std::size_t B = xs[0], C = xs[1], H = xs[2], W = xs[3];
    const std::size_t H_out = yshape[2], W_out = yshape[3];

    const std::size_t BC = B * C;
    ag::parallel::parallel_for(BC, /*grain=*/1, [&](std::size_t bc0, std::size_t bc1){
      for (std::size_t bc = bc0; bc < bc1; ++bc) {
        const std::size_t b = bc / C;
        const std::size_t c = bc % C;
        const std::size_t base = b * xstr[0] + c * xstr[1];

        for (std::size_t oh = 0; oh < H_out; ++oh) {
          const std::size_t h0 = (oh * sH_ > pH_) ? (oh * sH_ - pH_) : 0;
          const std::size_t h1 = std::min(h0 + kH_, H);
          for (std::size_t ow = 0; ow < W_out; ++ow) {
            const std::size_t w0 = (ow * sW_ > pW_) ? (ow * sW_ - pW_) : 0;
            const std::size_t w1 = std::min(w0 + kW_, W);

            // first pass: count ties
            float m = -std::numeric_limits<float>::infinity();
            for (std::size_t ih = h0; ih < h1; ++ih) {
              const std::size_t row = base + ih * xstr[2];
              for (std::size_t iw = w0; iw < w1; ++iw) {
                const std::size_t xi = row + iw * xstr[3];
                m = std::max(m, Xn->value[xi]);
              }
            }
            std::size_t ties = 0;
            for (std::size_t ih = h0; ih < h1; ++ih) {
              const std::size_t row = base + ih * xstr[2];
              for (std::size_t iw = w0; iw < w1; ++iw) {
                const std::size_t xi = row + iw * xstr[3];
                if (Xn->value[xi] == m) ++ties;
              }
            }
            if (ties == 0) continue;

            const std::size_t yi = b * ystr[0] + c * ystr[1] + oh * ystr[2] + ow * ystr[3];
            const float g = o->grad[yi] / float(ties);

            for (std::size_t ih = h0; ih < h1; ++ih) {
              const std::size_t row = base + ih * xstr[2];
              for (std::size_t iw = w0; iw < w1; ++iw) {
                const std::size_t xi = row + iw * xstr[3];
                if (Xn->value[xi] == m) Xn->grad[xi] += g;
              }
            }
          }
        }
      }
    });
  };


  return make_from_node(out);
}

Variable AvgPool2d::forward(const Variable& X) {
  const auto& xs = X.n->shape; // [B,C,H,W]
  if (xs.size()!=4) throw std::invalid_argument("AvgPool2d expects NCHW");
  std::size_t B=xs[0], C=xs[1], H=xs[2], W=xs[3];
  std::size_t H_out = safe_out_dim(H, kH, sH, pH);
  std::size_t W_out = safe_out_dim(W, kW, sW, pW);
  std::vector<std::size_t> yshape = {B,C,H_out,W_out};

  auto out = std::make_shared<Node>();
  out->shape = yshape;
  out->value.assign(numel(yshape), 0.0f);
  if (X.n->requires_grad) out->grad.assign(numel(yshape), 0.0f);
  out->requires_grad = X.n->requires_grad;
  out->parents = {X.n};

  const auto xstr = strides_for(xs);
  const auto ystr = strides_for(yshape);

  // forward: average (parallel over outputs)
  const std::size_t outN = numel(yshape);
  const std::size_t OUT_GRAIN = 1024;
  ag::parallel::parallel_for(outN, OUT_GRAIN, [&](std::size_t i0, std::size_t i1){
    const std::size_t strideB = C * H_out * W_out;
    const std::size_t strideC = H_out * W_out;
    const std::size_t strideH = W_out;
    for (std::size_t lin = i0; lin < i1; ++lin) {
      std::size_t tmp = lin;
      const std::size_t b = tmp / strideB; tmp %= strideB;
      const std::size_t c = tmp / strideC; tmp %= strideC;
      const std::size_t oh = tmp / strideH;
      const std::size_t ow = tmp % strideH;

      const std::size_t base = b * xstr[0] + c * xstr[1];
      const std::size_t h0 = (oh * sH > pH) ? (oh * sH - pH) : 0;
      const std::size_t h1 = std::min(h0 + kH, H);
      const std::size_t w0 = (ow * sW > pW) ? (ow * sW - pW) : 0;
      const std::size_t w1 = std::min(w0 + kW, W);

      float acc = 0.0f;
      std::size_t count = 0;
      for (std::size_t ih = h0; ih < h1; ++ih) {
        const std::size_t row = base + ih * xstr[2];
        for (std::size_t iw = w0; iw < w1; ++iw) {
          const std::size_t xi = row + iw * xstr[3];
          acc += X.n->value[xi];
          ++count;
        }
      }
      out->value[lin] = (count == 0) ? 0.0f : acc / float(count);
    }
  });

  out->backward = [kH_=this->kH, kW_=this->kW, sH_=this->sH, sW_=this->sW,
                   pH_=this->pH, pW_=this->pW,
                   Xn = X.n, xs, yshape,
                   oweak = std::weak_ptr<ag::Node>(out)]() {
    auto op = oweak.lock(); if (!op) return;
    ag::Node* o = op.get();

    if (!Xn || !Xn->requires_grad) return;
    if (Xn->grad.size() != Xn->value.size()) Xn->grad.assign(Xn->value.size(), 0.0f);

    const auto xstr = strides_for(xs);
    const auto ystr = strides_for(yshape);
    const std::size_t B = xs[0], C = xs[1], H = xs[2], W = xs[3];
    const std::size_t H_out = yshape[2], W_out = yshape[3];

    const std::size_t BC = B * C;
    ag::parallel::parallel_for(BC, /*grain=*/1, [&](std::size_t bc0, std::size_t bc1){
      for (std::size_t bc = bc0; bc < bc1; ++bc) {
        const std::size_t b = bc / C;
        const std::size_t c = bc % C;
        const std::size_t base = b * xstr[0] + c * xstr[1];

        for (std::size_t oh = 0; oh < H_out; ++oh) {
          const std::size_t h0 = (oh * sH_ > pH_) ? (oh * sH_ - pH_) : 0;
          const std::size_t h1 = std::min(h0 + kH_, H);
          for (std::size_t ow = 0; ow < W_out; ++ow) {
            const std::size_t w0 = (ow * sW_ > pW_) ? (ow * sW_ - pW_) : 0;
            const std::size_t w1 = std::min(w0 + kW_, W);
            const std::size_t cnt = (h1 - h0) * (w1 - w0);
            if (cnt == 0) continue;
            const std::size_t yi = b * ystr[0] + c * ystr[1] + oh * ystr[2] + ow * ystr[3];
            const float g = o->grad[yi] / float(cnt);
            for (std::size_t ih = h0; ih < h1; ++ih) {
              const std::size_t row = base + ih * xstr[2];
              for (std::size_t iw = w0; iw < w1; ++iw) {
                const std::size_t xi = row + iw * xstr[3];
                Xn->grad[xi] += g;
              }
            }
          }
        }
      }
    });
  };

  return make_from_node(out);
}

} // namespace ag::nn
