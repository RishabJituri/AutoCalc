#include "ag/nn/layers/normalization.hpp"
#include "ag/ops/tensor_utils.hpp"
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
  // Optional (nice for checkpoints / introspection)
  register_parameter("gamma", gamma);
  register_parameter("beta",  beta);
}

Variable BatchNorm2d::forward(const Variable& X) {
  const auto& xs = X.n->shape; // [B,C,H,W]
  if (xs.size() != 4 || xs[1] != C)
    throw std::invalid_argument("BatchNorm2d: expect [B,C,H,W] with matching C");

  std::size_t B = xs[0], H = xs[2], W = xs[3];
  std::size_t NHW = B * H * W;

  // compute per-channel mean/var over N*H*W
  std::vector<float> mean(C, 0.0f), var(C, 0.0f);
  const auto xstr = strides_for(xs);

  for (std::size_t c = 0; c < C; ++c) {
    float m = 0.0f;
    for (std::size_t b=0; b<B; ++b)
    for (std::size_t h=0; h<H; ++h)
    for (std::size_t w=0; w<W; ++w) {
      std::size_t i = b*xstr[0] + c*xstr[1] + h*xstr[2] + w*xstr[3];
      m += X.n->value[i];
    }
    m /= float(NHW);
    mean[c] = m;

    float vv = 0.0f;
    for (std::size_t b=0; b<B; ++b)
    for (std::size_t h=0; h<H; ++h)
    for (std::size_t w=0; w<W; ++w) {
      std::size_t i = b*xstr[0] + c*xstr[1] + h*xstr[2] + w*xstr[3];
      float d = X.n->value[i] - m;
      vv += d * d;
    }
    var[c] = vv / float(NHW);
  }

  // update or use running stats
  if (training()) {
    for (std::size_t c=0; c<C; ++c) {
      // PyTorch-style: new = (1-m)*old + m*batch
      running_mean.n->value[c] = (1.0f - momentum) * running_mean.n->value[c] + momentum * mean[c];
      running_var .n->value[c] = (1.0f - momentum) * running_var .n->value[c] + momentum * var[c];
    }
  } else {
    mean = running_mean.n->value;
    var  = running_var .n->value;
  }

  // output node
  auto out = std::make_shared<Node>();
  out->shape = X.n->shape;
  out->value.resize(numel(out->shape));
  out->grad.assign(numel(out->shape), 0.0f);
  out->requires_grad = (X.n->requires_grad || gamma.n->requires_grad || beta.n->requires_grad);
  out->parents = { X.n, gamma.n, beta.n };

  // forward normalize
  std::vector<float> inv_std(C, 0.0f);
  for (std::size_t c=0; c<C; ++c) inv_std[c] = 1.0f / std::sqrt(var[c] + eps);

  for (std::size_t b=0; b<B; ++b)
  for (std::size_t c=0; c<C; ++c)
  for (std::size_t h=0; h<H; ++h)
  for (std::size_t w=0; w<W; ++w) {
    std::size_t i = b*xstr[0] + c*xstr[1] + h*xstr[2] + w*xstr[3];
    float xhat = (X.n->value[i] - mean[c]) * inv_std[c];
    out->value[i] = xhat * gamma.n->value[c] + beta.n->value[c];
  }

  // backward (zero-arg lambda; get o via weak_ptr)
out->backward = [this, Xn = X.n, Gn = gamma.n, Bn = beta.n, xs, mean, inv_std, NHW,
                 oweak = std::weak_ptr<ag::Node>(out)]() {
  auto op = oweak.lock(); if (!op) return;
  ag::Node* o = op.get();

  const auto xstr_local = strides_for(xs);

  // d gamma
  if (Gn && Gn->requires_grad) {
    std::vector<float> dgamma(this->C, 0.0f);
    for (std::size_t b=0; b<xs[0]; ++b)
    for (std::size_t c=0; c<this->C; ++c)
    for (std::size_t h=0; h<xs[2]; ++h)
    for (std::size_t w=0; w<xs[3]; ++w) {
      std::size_t i = b*xstr_local[0] + c*xstr_local[1] + h*xstr_local[2] + w*xstr_local[3];
      float xhat = (Xn->value[i] - mean[c]) * inv_std[c];
      dgamma[c] += o->grad[i] * xhat;
    }
    for (std::size_t c=0; c<this->C; ++c) Gn->grad[c] += dgamma[c];
  }

  // d beta
  if (Bn && Bn->requires_grad) {
    std::vector<float> dbeta(this->C, 0.0f);
    for (std::size_t b=0; b<xs[0]; ++b)
    for (std::size_t c=0; c<this->C; ++c)
    for (std::size_t h=0; h<xs[2]; ++h)
    for (std::size_t w=0; w<xs[3]; ++w) {
      std::size_t i = b*xstr_local[0] + c*xstr_local[1] + h*xstr_local[2] + w*xstr_local[3];
      dbeta[c] += o->grad[i];
    }
    for (std::size_t c=0; c<this->C; ++c) Bn->grad[c] += dbeta[c];
  }

  // d x
  if (Xn && Xn->requires_grad) {
    std::vector<float> sum_dy(this->C, 0.0f), sum_dy_xhat(this->C, 0.0f);

    for (std::size_t b=0; b<xs[0]; ++b)
    for (std::size_t c=0; c<this->C; ++c)
    for (std::size_t h=0; h<xs[2]; ++h)
    for (std::size_t w=0; w<xs[3]; ++w) {
      std::size_t i = b*xstr_local[0] + c*xstr_local[1] + h*xstr_local[2] + w*xstr_local[3];
      float xhat = (Xn->value[i] - mean[c]) * inv_std[c];
      sum_dy[c]      += o->grad[i];
      sum_dy_xhat[c] += o->grad[i] * xhat;
    }

    for (std::size_t b=0; b<xs[0]; ++b)
    for (std::size_t c=0; c<this->C; ++c)
    for (std::size_t h=0; h<xs[2]; ++h)
    for (std::size_t w=0; w<xs[3]; ++w) {
      std::size_t i = b*xstr_local[0] + c*xstr_local[1] + h*xstr_local[2] + w*xstr_local[3];
      float xhat = (Xn->value[i] - mean[c]) * inv_std[c];
      float ggamma = (Gn ? Gn->value[c] : 1.0f);
      float term = float(NHW) * o->grad[i] - sum_dy[c] - xhat * sum_dy_xhat[c];
      Xn->grad[i] += (ggamma * inv_std[c] / float(NHW)) * term;
    }
  }
};

  return make_from_node(out);
}

} // namespace ag::nn
