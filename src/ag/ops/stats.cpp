#include "ag/ops/stats.hpp"
#include "ag/ops/reduce.hpp"
#include "ag/ops/elementwise.hpp"
#include "ag/ops/activations.hpp"
#include "ag/ops/tensor_utils.hpp"
#include <cmath>
#include <stdexcept>
#include <algorithm>
#include <limits>   
#include <memory>  

namespace ag {
using detail::numel;
using detail::strides_for;
using detail::unravel_index;
using detail::ravel_index;

static std::vector<std::size_t> normalize_axes(const std::vector<int>& axes_in,
                                               std::size_t rank) {
  if (axes_in.empty()) {
    std::vector<std::size_t> all(rank);
    for (std::size_t i=0;i<rank;++i) all[i]=i;
    return all;
  }
  std::vector<std::size_t> axes;
  axes.reserve(axes_in.size());
  for (int a : axes_in) {
    int aa = a < 0 ? (int)rank + a : a;
    if (aa < 0 || aa >= (int)rank) throw std::invalid_argument("reduce_max: axis out of range");
    axes.push_back((std::size_t)aa);
  }
  std::sort(axes.begin(), axes.end());
  axes.erase(std::unique(axes.begin(), axes.end()), axes.end());
  return axes;
}

Variable reduce_max(const Variable& X, const std::vector<int>& axes_in, bool keepdims) {
  const auto& shp = X.n->shape;
  const std::size_t rank = shp.size();
  auto axes = normalize_axes(axes_in, rank);

  // Build output shape
  std::vector<std::size_t> out_shape = shp;
  if (axes.empty()) {
    out_shape = keepdims ? std::vector<std::size_t>(rank, 1) : std::vector<std::size_t>{};
  } else {
    if (keepdims) {
      for (auto a : axes) out_shape[a] = 1;
    } else {
      std::vector<bool> keep(rank, true);
      for (auto a : axes) keep[a] = false;
      std::vector<std::size_t> tmp;
      for (std::size_t i=0;i<rank;++i) if (keep[i]) tmp.push_back(shp[i]);
      out_shape = tmp;
    }
  }

  // Iterate all elements and compute max per reduced group
  auto out = std::make_shared<Node>();
  out->shape = out_shape;
  out->value.assign(numel(out_shape), -std::numeric_limits<float>::infinity());
  out->grad.assign(numel(out_shape), 0.0f);
  out->requires_grad = X.n->requires_grad;
  out->parents = {X.n};

  const auto xstr = strides_for(shp);
  const auto ostr = strides_for(out_shape);

  auto map_out_index = [&](const std::vector<std::size_t>& idx)->std::size_t{
    if (axes.empty()) return 0;
    if (keepdims) {
      std::vector<std::size_t> oidx = idx;
      for (auto a: axes) oidx[a] = 0;
      return ravel_index(oidx, ostr);
    } else {
      std::vector<std::size_t> oidx;
      oidx.reserve(idx.size());
      for (std::size_t i=0;i<idx.size();++i) {
        if (std::find(axes.begin(), axes.end(), i) == axes.end()) oidx.push_back(idx[i]);
      }
      if (oidx.empty()) return 0;
      return ravel_index(oidx, ostr);
    }
  };

  // forward
  for (std::size_t lin=0; lin<numel(shp); ++lin) {
    auto idx = unravel_index(lin, shp);
    std::size_t oi = map_out_index(idx);
    float v = X.n->value[lin];
    if (v > out->value[oi]) out->value[oi] = v;
  }

  // backward
  out->backward = [Xn = X.n, shp, out_shape, axes, ostr,
                 oweak = std::weak_ptr<ag::Node>(out)]() {
  auto op = oweak.lock(); if (!op) return;
  ag::Node* o = op.get();
  if (!Xn || !Xn->requires_grad) return;

  auto map_out_index = [&](const std::vector<std::size_t>& idx)->std::size_t{
    if (axes.empty()) return 0;
    if (o->shape.size() == shp.size()) {
      std::vector<std::size_t> oidx = idx;
      for (auto a: axes) oidx[a] = 0;
      return ravel_index(oidx, ostr);
    } else {
      std::vector<std::size_t> oidx;
      oidx.reserve(idx.size());
      for (std::size_t i=0;i<idx.size();++i)
        if (std::find(axes.begin(), axes.end(), i) == axes.end()) oidx.push_back(idx[i]);
      return oidx.empty() ? 0 : ravel_index(oidx, ostr);
    }
  };

  // For each input element, find its output group and split grad among ties.
  for (std::size_t lin = 0; lin < numel(shp); ++lin) {
    auto idx = unravel_index(lin, shp);
    std::size_t oi = map_out_index(idx);
    float m = o->value[oi];

    // count ties in this group
    std::size_t ties = 0;
    for (std::size_t lin2 = 0; lin2 < numel(shp); ++lin2) {
      if (map_out_index(unravel_index(lin2, shp)) == oi &&
          Xn->value[lin2] == m) {
        ++ties;
      }
    }
    if (ties > 0 && Xn->value[lin] == m) {
      Xn->grad[lin] += o->grad[oi] / float(ties);
    }
  }
};


  return make_from_node(out);
}

Variable logsumexp(const Variable& X, const std::vector<int>& axes, bool keepdims) {
  auto M = reduce_max(X, axes, true);
  auto Xm = sub(X, M);           // broadcasted
  auto E = expv(Xm);
  auto S = reduce_sum(E, axes, keepdims);
  auto L = logv(S);
  if (keepdims) {
    // result already includes max in kept dims
    return add(L, (keepdims? M : reduce_sum(M, std::vector<int>{}, false)));
  } else {
    // need to squeeze M to match L for +; reduce M over same axes without keepdims
    auto M2 = reduce_max(X, axes, false);
    return add(L, M2);
  }
}

Variable softmax(const Variable& X, int axis) {
  int rank = (int)X.n->shape.size();
  int ax = axis < 0 ? rank + axis : axis;
  if (ax < 0 || ax >= rank) throw std::invalid_argument("softmax: axis out of range");
  // lse over given axis
  std::vector<int> axes = {ax};
  auto M = reduce_max(X, axes, true);
  auto E = expv(sub(X, M));
  auto S = reduce_sum(E, axes, true);
  return div(E, S); // broadcasted division
}

std::vector<std::size_t> argmax_lastdim(const Variable& X) {
  const auto& shp = X.n->shape;
  if (shp.empty()) return {};
  std::size_t B = 1;
  for (std::size_t i=0; i+1<shp.size(); ++i) B *= shp[i];
  std::size_t C = shp.back();
  std::vector<std::size_t> out(B, 0);
  for (std::size_t b=0; b<B; ++b) {
    float best = -1e30f;
    std::size_t idx = 0;
    for (std::size_t c=0; c<C; ++c) {
      float v = X.n->value[b*C + c];
      if (v > best) { best = v; idx = c; }
    }
    out[b] = idx;
  }
  return out;
}

} // namespace ag
