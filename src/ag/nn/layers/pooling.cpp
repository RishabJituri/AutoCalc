#include "ag/nn/layers/pooling.hpp"
#include "ag/ops/tensor_utils.hpp"
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
  out->value.assign(numel(yshape), -std::numeric_limits<double>::infinity());
  out->grad.assign(numel(yshape), 0.0);
  out->requires_grad = X.n->requires_grad;
  out->parents = {X.n};

  const auto xstr = strides_for(xs);
  const auto ystr = strides_for(yshape);

  // forward: compute max with padding treated as -inf
  for (std::size_t b=0;b<B;++b)
  for (std::size_t c=0;c<C;++c)
  for (std::size_t oh=0;oh<H_out;++oh)
  for (std::size_t ow=0;ow<W_out;++ow) {
    double m = -std::numeric_limits<double>::infinity();
    for (std::size_t kh=0;kh<kH;++kh)
    for (std::size_t kw=0;kw<kW;++kw) {
      long long ih = (long long)oh*(long long)sH + (long long)kh - (long long)pH;
      long long iw = (long long)ow*(long long)sW + (long long)kw - (long long)pW;
      if (ih<0 || iw<0 || ih>=(long long)H || iw>=(long long)W) continue;
      std::size_t xi = b*xstr[0] + c*xstr[1] + (std::size_t)ih*xstr[2] + (std::size_t)iw*xstr[3];
      m = std::max(m, X.n->value[xi]);
    }
    out->value[b*ystr[0] + c*ystr[1] + oh*ystr[2] + ow*ystr[3]] = m;
  }

out->backward = [this, Xn = X.n, xs, yshape,
                 oweak = std::weak_ptr<ag::Node>(out)]() {
  auto op = oweak.lock(); if (!op) return;
  ag::Node* o = op.get();

  if (!Xn || !Xn->requires_grad) return;
  const auto xstr = strides_for(xs);
  const auto ystr = strides_for(yshape);
  std::size_t B = xs[0], C = xs[1], H = xs[2], W = xs[3];
  std::size_t H_out = yshape[2], W_out = yshape[3];

  for (std::size_t b=0;b<B;++b)
  for (std::size_t c=0;c<C;++c)
  for (std::size_t oh=0;oh<H_out;++oh)
  for (std::size_t ow=0;ow<W_out;++ow) {
    double m = o->value[b*ystr[0] + c*ystr[1] + oh*ystr[2] + ow*ystr[3]];
    std::size_t ties = 0;
    for (std::size_t kh=0; kh<this->kH; ++kh)
    for (std::size_t kw=0; kw<this->kW; ++kw) {
      long long ih = (long long)oh*(long long)this->sH + (long long)kh - (long long)this->pH;
      long long iw = (long long)ow*(long long)this->sW + (long long)kw - (long long)this->pW;
      if (ih<0 || iw<0 || ih>=(long long)H || iw>=(long long)W) continue;
      std::size_t xi = b*xstr[0] + c*xstr[1] + (std::size_t)ih*xstr[2] + (std::size_t)iw*xstr[3];
      if (Xn->value[xi] == m) ++ties;
    }
    if (ties == 0) continue;
    double g = o->grad[b*ystr[0] + c*ystr[1] + oh*ystr[2] + ow*ystr[3]] / double(ties);
    for (std::size_t kh=0; kh<this->kH; ++kh)
    for (std::size_t kw=0; kw<this->kW; ++kw) {
      long long ih = (long long)oh*(long long)this->sH + (long long)kh - (long long)this->pH;
      long long iw = (long long)ow*(long long)this->sW + (long long)kw - (long long)this->pW;
      if (ih<0 || iw<0 || ih>=(long long)H || iw>=(long long)W) continue;
      std::size_t xi = b*xstr[0] + c*xstr[1] + (std::size_t)ih*xstr[2] + (std::size_t)iw*xstr[3];
      if (Xn->value[xi] == m) Xn->grad[xi] += g;
    }
  }
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
  out->value.assign(numel(yshape), 0.0);
  out->grad.assign(numel(yshape), 0.0);
  out->requires_grad = X.n->requires_grad;
  out->parents = {X.n};

  const auto xstr = strides_for(xs);
  const auto ystr = strides_for(yshape);

  // forward: average
  for (std::size_t b=0;b<B;++b)
  for (std::size_t c=0;c<C;++c)
  for (std::size_t oh=0;oh<H_out;++oh)
  for (std::size_t ow=0;ow<W_out;++ow) {
    double acc = 0.0;
    std::size_t count = 0;
    for (std::size_t kh=0;kh<kH;++kh)
    for (std::size_t kw=0;kw<kW;++kw) {
      long long ih = (long long)oh*(long long)sH + (long long)kh - (long long)pH;
      long long iw = (long long)ow*(long long)sW + (long long)kw - (long long)pW;
      if (ih<0 || iw<0 || ih>=(long long)H || iw>=(long long)W) continue;
      std::size_t xi = b*xstr[0] + c*xstr[1] + (std::size_t)ih*xstr[2] + (std::size_t)iw*xstr[3];
      acc += X.n->value[xi];
      ++count;
    }
    std::size_t yi = b*ystr[0] + c*ystr[1] + oh*ystr[2] + ow*ystr[3];
    out->value[yi] = (count == 0) ? 0.0 : acc / double(count);
  }

  out->backward = [this, Xn = X.n, xs, yshape,
                   oweak = std::weak_ptr<ag::Node>(out)]() {
    auto op = oweak.lock(); if (!op) return;
    ag::Node* o = op.get();

    if (!Xn || !Xn->requires_grad) return;
    const auto xstr = strides_for(xs);
    const auto ystr = strides_for(yshape);
    std::size_t B = xs[0], C = xs[1], H = xs[2], W = xs[3];
    std::size_t H_out = yshape[2], W_out = yshape[3];

    for (std::size_t b=0;b<B;++b)
    for (std::size_t c=0;c<C;++c)
    for (std::size_t oh=0;oh<H_out;++oh)
    for (std::size_t ow=0;ow<W_out;++ow) {
      std::size_t count = 0;
      for (std::size_t kh=0; kh<this->kH; ++kh)
      for (std::size_t kw=0; kw<this->kW; ++kw) {   // <-- fixed
        long long ih = (long long)oh*(long long)this->sH + (long long)kh - (long long)this->pH;
        long long iw = (long long)ow*(long long)this->sW + (long long)kw - (long long)this->pW;
        if (ih<0 || iw<0 || ih>=(long long)H || iw>=(long long)W) continue;
        ++count;
      }
      if (count==0) continue;
      double g = o->grad[b*ystr[0] + c*ystr[1] + oh*ystr[2] + ow*ystr[3]] / double(count);
      for (std::size_t kh=0; kh<this->kH; ++kh)
      for (std::size_t kw=0; kw<this->kW; ++kw) {   // <-- fixed
        long long ih = (long long)oh*(long long)this->sH + (long long)kh - (long long)this->pH;
        long long iw = (long long)ow*(long long)this->sW + (long long)kw - (long long)this->pW;
        if (ih<0 || iw<0 || ih>=(long long)H || iw>=(long long)W) continue;
        std::size_t xi = b*xstr[0] + c*xstr[1] + (std::size_t)ih*xstr[2] + (std::size_t)iw*xstr[3];
        Xn->grad[xi] += g;
      }
    }
  };

  return make_from_node(out);
}


} // namespace ag::nn
