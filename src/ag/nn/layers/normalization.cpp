#include "ag/nn/layers/normalization.hpp"
#include "ag/ops/tensor_utils.hpp"
#include "ag/parallel/parallel_for.hpp"
#include <cmath>
#include <stdexcept>
#include <memory>   // for weak_ptr
#include <limits>

namespace ag::nn {
using ag::Variable;
using ag::Node;
using ag::detail::numel;
using ag::detail::strides_for;

BatchNorm2d::BatchNorm2d(std::size_t C, float eps, float momentum)
  : C(C), eps(eps), momentum(momentum),
    gamma(std::vector<float>(C, 1.0f), {C}, /*requires_grad=*/true),
    beta (std::vector<float>(C, 0.0f), {C}, /*requires_grad=*/true),
    running_mean(std::vector<float>(C, 0.0f), {C}, /*requires_grad=*/false),
    running_var (std::vector<float>(C, 1.0f), {C}, /*requires_grad=*/false)
{
  register_parameter("gamma", gamma);
  register_parameter("beta",  beta);
}

Variable BatchNorm2d::forward(const Variable& X) {
  const auto& xs = X.n->shape; // [B,C,H,W]
  if (xs.size() != 4 || xs[1] != C)
    throw std::invalid_argument("BatchNorm2d: expect [B,C,H,W] with matching C");

  std::size_t B = xs[0], H = xs[2], W = xs[3];
  const std::size_t NHW = B * H * W;

  // temporary per-channel stats
  std::vector<float> mean(C, 0.0f), var(C, 0.0f);
  const auto xstr = strides_for(xs);
  const auto Xv = X.n->value.data();

  // compute mean/var per-channel using Welford, parallel over channels
  const std::size_t CH_GRAIN = 1;
  ag::parallel::parallel_for(C, CH_GRAIN, [&](std::size_t c0, std::size_t c1){
    for (std::size_t c = c0; c < c1; ++c) {
      double m = 0.0, M2 = 0.0;
      std::size_t n = 0;
      const std::size_t base_c = c * xstr[1];
      for (std::size_t b = 0; b < B; ++b) {
        const std::size_t base_b = b * xstr[0] + base_c;
        for (std::size_t h = 0; h < H; ++h) {
          const std::size_t row = base_b + h * xstr[2];
          for (std::size_t w = 0; w < W; ++w) {
            const float x = Xv[row + w * xstr[3]];
            ++n;
            double delta = x - m;
            m += delta / double(n);
            M2 += delta * (x - m);
          }
        }
      }
      mean[c] = static_cast<float>(m);
      var[c] = static_cast<float>( (n > 0) ? (M2 / double(n)) : 0.0 );
    }
  });

  // update running stats
  if (training()) {
    for (std::size_t c = 0; c < C; ++c) {
      running_mean.n->value[c] = (1.0f - momentum) * running_mean.n->value[c] + momentum * mean[c];
      running_var .n->value[c] = (1.0f - momentum) * running_var .n->value[c] + momentum * var[c];
    }
  } else {
    mean = running_mean.n->value;
    var  = running_var .n->value;
  }

  // prepare output
  auto out = std::make_shared<Node>();
  out->shape = X.n->shape;
  out->value.resize(numel(out->shape));
  if (X.n->requires_grad || gamma.n->requires_grad || beta.n->requires_grad)
    out->grad.assign(numel(out->shape), 0.0f);
  out->requires_grad = (X.n->requires_grad || gamma.n->requires_grad || beta.n->requires_grad);
  out->parents = { X.n, gamma.n, beta.n };

  // compute inv_std per-channel
  std::vector<float> inv_std(C);
  for (std::size_t c = 0; c < C; ++c) inv_std[c] = 1.0f / std::sqrt(var[c] + eps);

  // forward normalize - parallel over channels
  const auto Gv = gamma.n->value.data();
  const auto Bv = beta.n->value.data();
  const std::size_t FWD_GRAIN = 1;
  ag::parallel::parallel_for(C, FWD_GRAIN, [&](std::size_t c0, std::size_t c1){
    for (std::size_t c = c0; c < c1; ++c) {
      const float g = Gv[c];
      const float bb = Bv[c];
      const float m = mean[c];
      const float s = inv_std[c];
      const std::size_t base_c = c * xstr[1];
      for (std::size_t b = 0; b < B; ++b) {
        const std::size_t base_b = b * xstr[0] + base_c;
        for (std::size_t h = 0; h < H; ++h) {
          const std::size_t row = base_b + h * xstr[2];
          for (std::size_t w = 0; w < W; ++w) {
            const std::size_t i = row + w * xstr[3];
            float xhat = (Xv[i] - m) * s;
            out->value[i] = xhat * g + bb;
          }
        }
      }
    }
  });

  // backward
  out->backward = [C_=this->C, Xn = X.n, Gn = gamma.n, Bn = beta.n, xs, mean, inv_std, NHW,
                   oweak = std::weak_ptr<ag::Node>(out)]() {
    auto op = oweak.lock(); if (!op) return;
    ag::Node* o = op.get();

    const auto xstr_local = strides_for(xs);
    const auto Xv_local = Xn->value.data();

    // dgamma / dbeta (single pass per-channel)
    if (Gn && Gn->requires_grad) {
      std::vector<float> dgamma(Gn->value.size(), 0.0f);
      const std::size_t CHG_GRAIN = 1;
      ag::parallel::parallel_for(Gn->value.size(), CHG_GRAIN, [&](std::size_t c0, std::size_t c1){
        for (std::size_t c = c0; c < c1; ++c) {
          double a = 0.0;
          const float m = mean[c];
          const float s = inv_std[c];
          const std::size_t base_c = c * xstr_local[1];
          for (std::size_t b = 0; b < xs[0]; ++b) {
            const std::size_t base_b = b * xstr_local[0] + base_c;
            for (std::size_t h = 0; h < xs[2]; ++h) {
              const std::size_t row = base_b + h * xstr_local[2];
              for (std::size_t w = 0; w < xs[3]; ++w) {
                const std::size_t i = row + w * xstr_local[3];
                const float xhat = (Xv_local[i] - m) * s;
                a += double(o->grad[i]) * double(xhat);
              }
            }
          }
          dgamma[c] = static_cast<float>(a);
        }
      });
      for (std::size_t c = 0; c < Gn->grad.size(); ++c) Gn->grad[c] += dgamma[c];
    }

    if (Bn && Bn->requires_grad) {
      std::vector<float> dbeta(Bn->value.size(), 0.0f);
      const std::size_t CHB_GRAIN = 1;
      ag::parallel::parallel_for(Bn->value.size(), CHB_GRAIN, [&](std::size_t c0, std::size_t c1){
        for (std::size_t c = c0; c < c1; ++c) {
          double a = 0.0;
          const std::size_t base_c = c * xstr_local[1];
          for (std::size_t b = 0; b < xs[0]; ++b) {
            const std::size_t base_b = b * xstr_local[0] + base_c;
            for (std::size_t h = 0; h < xs[2]; ++h) {
              const std::size_t row = base_b + h * xstr_local[2];
              for (std::size_t w = 0; w < xs[3]; ++w) {
                const std::size_t i = row + w * xstr_local[3];
                a += double(o->grad[i]);
              }
            }
          }
          dbeta[c] = static_cast<float>(a);
        }
      });
      for (std::size_t c = 0; c < Bn->grad.size(); ++c) Bn->grad[c] += dbeta[c];
    }

    // dX: two-pass per-channel
    if (Xn && Xn->requires_grad) {
      if (Xn->grad.size() != Xn->value.size()) Xn->grad.assign(Xn->value.size(), 0.0f);
      std::vector<double> sum_dy(C_, 0.0), sum_dy_xhat(C_, 0.0);
      const std::size_t CHS_GRAIN = 1;
      ag::parallel::parallel_for(C_, CHS_GRAIN, [&](std::size_t c0, std::size_t c1){
        for (std::size_t c = c0; c < c1; ++c) {
          double sdy = 0.0, sdyx = 0.0;
          const float m = mean[c];
          const float s = inv_std[c];
          const std::size_t base_c = c * xstr_local[1];
          for (std::size_t b = 0; b < xs[0]; ++b) {
            const std::size_t base_b = b * xstr_local[0] + base_c;
            for (std::size_t h = 0; h < xs[2]; ++h) {
              const std::size_t row = base_b + h * xstr_local[2];
              for (std::size_t w = 0; w < xs[3]; ++w) {
                const std::size_t i = row + w * xstr_local[3];
                const double dy = double(o->grad[i]);
                sdy += dy;
                const double xhat = double(Xv_local[i] - m) * double(s);
                sdyx += dy * xhat;
              }
            }
          }
          sum_dy[c] = sdy;
          sum_dy_xhat[c] = sdyx;
        }
      });

      // distribute
      const std::size_t CHD_GRAIN = 1;
      ag::parallel::parallel_for(C_, CHD_GRAIN, [&](std::size_t c0, std::size_t c1){
        for (std::size_t c = c0; c < c1; ++c) {
          const float m = mean[c];
          const float s = inv_std[c];
          const float ggamma = (Gn ? Gn->value[c] : 1.0f);
          const double sdy = sum_dy[c];
          const double sdyx = sum_dy_xhat[c];
          const double scale = double(ggamma) * double(s) / double(NHW);
          const std::size_t base_c = c * xstr_local[1];
          for (std::size_t b = 0; b < xs[0]; ++b) {
            const std::size_t base_b = b * xstr_local[0] + base_c;
            for (std::size_t h = 0; h < xs[2]; ++h) {
              const std::size_t row = base_b + h * xstr_local[2];
              for (std::size_t w = 0; w < xs[3]; ++w) {
                const std::size_t i = row + w * xstr_local[3];
                const double xhat = double(Xv_local[i] - m) * double(s);
                const double term = double(NHW) * double(o->grad[i]) - sdy - xhat * sdyx;
                Xn->grad[i] += static_cast<float>(scale * term);
              }
            }
          }
        }
      });
    }
  };

  return make_from_node(out);
}

std::vector<ag::Variable*> BatchNorm2d::_parameters() {
  std::vector<ag::Variable*> out;
  out.push_back(&gamma);
  out.push_back(&beta);
  return out;
}

} // namespace ag::nn
