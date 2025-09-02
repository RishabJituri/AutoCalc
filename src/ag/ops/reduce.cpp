#include "ag/ops/reduce.hpp"
#include "ag/ops/tensor_utils.hpp"
#include <unordered_set>
#include <algorithm>
#include <stdexcept>
#include <numeric>

namespace ag {
using detail::numel;
using detail::strides_for;
using detail::unravel_index;
using detail::ravel_index;

static std::vector<std::size_t> normalize_axes(const std::vector<int>& axes_in,
                                                std::size_t rank) {
  std::vector<std::size_t> axes;
  if (axes_in.empty()) {
    axes.resize(rank);
    std::iota(axes.begin(), axes.end(), std::size_t{0});
    return axes;
  }
  axes.reserve(axes_in.size());
  for (int ax : axes_in) {
    long a = ax;
    if (a < 0) a += long(rank);
    if (a < 0 || a >= long(rank))
      throw std::invalid_argument("reduce: axis out of range");
    axes.push_back(std::size_t(a));
  }
  std::sort(axes.begin(), axes.end());
  axes.erase(std::unique(axes.begin(), axes.end()), axes.end());
  return axes;
}

static std::vector<std::size_t>
reduced_shape(const std::vector<std::size_t>& in_shape,
              const std::vector<std::size_t>& axes,
              bool keepdims) {
  if (keepdims) {
    std::vector<std::size_t> s = in_shape;
    for (auto a : axes) s[a] = 1;
    return s;
  } else {
    std::vector<std::size_t> s;
    s.reserve(in_shape.size() - axes.size());
    std::size_t j = 0;
    for (std::size_t i = 0; i < in_shape.size(); ++i) {
      if (j < axes.size() && axes[j] == i) { ++j; continue; }
      s.push_back(in_shape[i]);
    }
    if (s.empty()) s = {1}; // represent scalar as [1]
    return s;
  }
}

static std::vector<std::size_t>
collapsed_index(const std::vector<std::size_t>& in_idx,
                const std::vector<std::size_t>& axes,
                bool keepdims) {
  if (keepdims) {
    std::vector<std::size_t> out = in_idx;
    for (auto a : axes) out[a] = 0;
    return out;
  } else {
    std::vector<std::size_t> out;
    out.reserve(in_idx.size() - axes.size());
    std::size_t j = 0;
    for (std::size_t i = 0; i < in_idx.size(); ++i) {
      if (j < axes.size() && axes[j] == i) { ++j; continue; }
      out.push_back(in_idx[i]);
    }
    if (out.empty()) out = {0};
    return out;
  }
}

Variable reduce_sum(const Variable& X, const std::vector<int>& axes_in, bool keepdims) {
  const auto& shp = X.n->shape;
  const std::size_t rank = shp.size();
  auto axes = normalize_axes(axes_in, rank);

  const auto out_shape = reduced_shape(shp, axes, keepdims);
  const auto outN = numel(out_shape);
  const auto out_strides = strides_for(out_shape);
  const auto inN = numel(shp);

  auto out = std::make_shared<Node>();
  out->shape = out_shape;
  out->value.assign(outN, 0.0);
  out->grad.assign(outN, 0.0);
  out->requires_grad = X.n->requires_grad;
  out->parents = {X.n};

  // Forward accumulate
  for (std::size_t lin = 0; lin < inN; ++lin) {
    auto idx = unravel_index(lin, shp);
    auto cidx = collapsed_index(idx, axes, keepdims);
    auto olin = ravel_index(cidx, out_strides);
    out->value[olin] += X.n->value[lin];
  }

  std::weak_ptr<Node> ow = out, xw = X.n;
  out->backward = [ow, xw, shp, axes, keepdims, out_strides]() {
    auto o = ow.lock(); if (!o) return; auto x = xw.lock();
    if (x && x->requires_grad) {
      const auto inN = numel(shp);
      for (std::size_t lin = 0; lin < inN; ++lin) {
        auto idx = unravel_index(lin, shp);
        auto cidx = collapsed_index(idx, axes, keepdims);
        auto olin = ravel_index(cidx, out_strides);
        x->grad[lin] += o->grad[olin]; // d(sum)/dx = 1
      }
    }
  };

  return make_from_node(out);
}

Variable reduce_mean(const Variable& X, const std::vector<int>& axes_in, bool keepdims) {
  const auto& shp = X.n->shape;
  const std::size_t rank = shp.size();
  auto axes = normalize_axes(axes_in, rank);

  // compute number of elements reduced into each output element
  std::size_t red_count = 1;
  for (auto a : axes) red_count *= shp[a];

  auto Y = reduce_sum(X, axes_in, keepdims);
  for (auto& v : Y.n->value) v /= double(red_count);

  // Backward: scale incoming grad by 1/red_count before reuse of sum backward
  std::weak_ptr<Node> yw = Y.n;
  auto prev_backward = Y.n->backward;
  Y.n->backward = [yw, prev_backward, red_count]() {
    auto y = yw.lock(); if (!y) return;
    const double scale = 1.0 / double(red_count);
    for (auto& g : y->grad) g *= scale;
    if (prev_backward) prev_backward();
  };
  return Y;
}

Variable broadcast_to(const Variable& X, const std::vector<std::size_t>& out_shape) {
  // Validate broadcast compatibility (NumPy right-aligned)
  const auto& in_shape = X.n->shape;
  const std::size_t ra = in_shape.size(), rb = out_shape.size();
  for (std::size_t i = 0; i < rb; ++i) {
    const std::size_t ad = (i < rb - ra) ? 1 : in_shape[i - (rb - ra)];
    const std::size_t bd = out_shape[i];
    if (ad != bd && ad != 1)
      throw std::invalid_argument("broadcast_to: incompatible shapes");
  }

  const auto outN = numel(out_shape);
  auto out = std::make_shared<Node>();
  out->shape = out_shape;
  out->value.assign(outN, 0.0);
  out->grad.assign(outN, 0.0);
  out->requires_grad = X.n->requires_grad;
  out->parents = {X.n};

  const auto out_strides = strides_for(out_shape);

  // Forward: map each out index to an in index (use 0 on broadcasted dims)
  for (std::size_t lin = 0; lin < outN; ++lin) {
    auto oidx = unravel_index(lin, out_shape);
    std::vector<std::size_t> iidx;
    iidx.reserve(in_shape.size());
    const std::size_t rb2 = out_shape.size(), ra2 = in_shape.size();
    for (std::size_t i = 0; i < ra2; ++i) {
      // align from the right
      std::size_t od = oidx[i + (rb2 - ra2)];
      std::size_t ad = in_shape[i];
      iidx.push_back(ad == 1 ? 0 : od);
    }
    const auto istrides = strides_for(in_shape);
    auto ilin = ravel_index(iidx, istrides);
    out->value[lin] = X.n->value[ilin];
  }

  std::weak_ptr<Node> ow = out, xw = X.n;
  out->backward = [ow, xw, out_shape, in_shape]() {
    auto o = ow.lock(); if (!o) return; auto x = xw.lock();
    if (!x || !x->requires_grad) return;

    const auto outN = numel(out_shape);
    const auto istrides = strides_for(in_shape);

    for (std::size_t lin = 0; lin < outN; ++lin) {
      auto oidx = unravel_index(lin, out_shape);
      std::vector<std::size_t> iidx;
      iidx.reserve(in_shape.size());
      const std::size_t rb2 = out_shape.size(), ra2 = in_shape.size();
      for (std::size_t i = 0; i < ra2; ++i) {
        std::size_t od = oidx[i + (rb2 - ra2)];
        std::size_t ad = in_shape[i];
        iidx.push_back(ad == 1 ? 0 : od);
      }
      auto ilin = ravel_index(iidx, istrides);
      x->grad[ilin] += o->grad[lin]; // sum over broadcasted positions
    }
  };

  return make_from_node(out);
}

} // namespace ag
