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
#include "ag/parallel/parallel_for.hpp"

namespace ag {
using detail::numel;
using detail::strides_for;
using detail::unravel_index;
using detail::ravel_index;
using ag::parallel::parallel_for;

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

  // Per-output-group parallel max (scan only the reduced axes per output)
  const std::size_t outN = numel(out_shape);
  // Prepare reduced-dim info
  const std::size_t R = axes.size();
  std::vector<std::size_t> red_dims = axes; // reduced dims
  std::vector<std::size_t> red_extents(R), red_strides(R);
  std::size_t reduced_count = 1;
  for (std::size_t r = 0; r < R; ++r) {
    red_extents[r] = shp[red_dims[r]];
    red_strides[r] = xstr[red_dims[r]];
    reduced_count *= red_extents[r];
  }

  const std::size_t GROUP_GRAIN = 64; // tuneable
  parallel_for(outN, GROUP_GRAIN, [&](std::size_t o0, std::size_t o1){
    std::vector<std::size_t> out_idx; out_idx.reserve(out_shape.size());
    std::vector<std::size_t> full_idx(shp.size());
    std::vector<std::size_t> red_coord(R);
    for (std::size_t oi = o0; oi < o1; ++oi) {
      // build full_idx (length rank) by mapping output coords into kept dims
      if (R == 0) {
        // full index runs over all dims
        for (std::size_t i = 0; i < shp.size(); ++i) full_idx[i] = 0;
      } else if (keepdims) {
        out_idx = unravel_index(oi, out_shape);
        // out_idx already length rank (keeps reduced dims as 0)
        for (std::size_t i = 0; i < shp.size(); ++i) full_idx[i] = out_idx[i];
      } else {
        out_idx = unravel_index(oi, out_shape);
        std::size_t p = 0;
        for (std::size_t i = 0; i < shp.size(); ++i) {
          if (std::find(red_dims.begin(), red_dims.end(), i) != red_dims.end()) {
            full_idx[i] = 0;
          } else {
            full_idx[i] = out_idx[p++];
          }
        }
      }

      // base linear index into X for this group's first combination (all reduced coords = 0)
      std::size_t base = ravel_index(full_idx, xstr);

      // iterate over reduced axes via multi-index to find max
      float best = -std::numeric_limits<float>::infinity();
      // reset red_coord
      for (std::size_t r = 0; r < R; ++r) { red_coord[r] = 0; }
      std::size_t offset = base;
      for (std::size_t rc = 0; rc < std::max<std::size_t>(std::size_t(1), reduced_count); ++rc) {
        float v = X.n->value[offset];
        if (v > best) best = v;

        // increment red_coord and offset
        for (int d = int(R) - 1; d >= 0; --d) {
          red_coord[d]++;
          offset += red_strides[d];
          if (red_coord[d] < red_extents[d]) break;
          // wrap
          offset -= red_strides[d] * red_extents[d];
          red_coord[d] = 0;
        }
      }
      out->value[oi] = best;
    }
  });

  // backward: for each output group, count ties then distribute o->grad[oi]/ties to matching inputs
  out->backward = [Xn = X.n, shp, out_shape, axes, ostr, oweak = std::weak_ptr<ag::Node>(out), xstr, keepdims]() {
    auto op = oweak.lock(); if (!op) return; ag::Node* o = op.get();
    if (!Xn || !Xn->requires_grad) return;

    if (Xn->grad.size() != Xn->value.size()) Xn->grad.assign(Xn->value.size(), 0.0f);

    const std::size_t outN2 = numel(out_shape);
    const std::size_t R2 = axes.size();
    std::vector<std::size_t> red_dims2 = axes;
    std::vector<std::size_t> red_extents2(R2), red_strides2(R2);
    std::size_t reduced_count2 = 1;
    for (std::size_t r = 0; r < R2; ++r) {
      red_extents2[r] = shp[red_dims2[r]];
      red_strides2[r] = xstr[red_dims2[r]];
      reduced_count2 *= red_extents2[r];
    }

    const std::size_t GROUP_GRAIN2 = 64;
    parallel_for(outN2, GROUP_GRAIN2, [&](std::size_t o0, std::size_t o1){
      std::vector<std::size_t> out_idx; out_idx.reserve(out_shape.size());
      std::vector<std::size_t> full_idx(shp.size());
      std::vector<std::size_t> red_coord(R2);
      for (std::size_t oi = o0; oi < o1; ++oi) {
        // build full_idx as in forward
        if (R2 == 0) {
          for (std::size_t i = 0; i < shp.size(); ++i) full_idx[i] = 0;
        } else if (keepdims) {
          out_idx = unravel_index(oi, out_shape);
          for (std::size_t i = 0; i < shp.size(); ++i) full_idx[i] = out_idx[i];
        } else {
          out_idx = unravel_index(oi, out_shape);
          std::size_t p = 0;
          for (std::size_t i = 0; i < shp.size(); ++i) {
            if (std::find(red_dims2.begin(), red_dims2.end(), i) != red_dims2.end()) {
              full_idx[i] = 0;
            } else {
              full_idx[i] = out_idx[p++];
            }
          }
        }

        std::size_t base = ravel_index(full_idx, xstr);
        // first pass: count ties
        std::size_t ties = 0;
        for (std::size_t r = 0; r < R2; ++r) red_coord[r] = 0;
        std::size_t offset = base;
        const float m = o->value[oi];
        for (std::size_t rc = 0; rc < std::max<std::size_t>(std::size_t(1), reduced_count2); ++rc) {
          if (Xn->value[offset] == m) ++ties;
          for (int d = int(R2) - 1; d >= 0; --d) {
            red_coord[d]++;
            offset += red_strides2[d];
            if (red_coord[d] < red_extents2[d]) break;
            offset -= red_strides2[d] * red_extents2[d];
            red_coord[d] = 0;
          }
        }
        if (ties == 0) continue;
        // second pass: distribute gradient
        const float share = static_cast<float>(o->grad[oi]) / float(ties);
        for (std::size_t r = 0; r < R2; ++r) red_coord[r] = 0;
        offset = base;
        for (std::size_t rc = 0; rc < std::max<std::size_t>(std::size_t(1), reduced_count2); ++rc) {
          if (Xn->value[offset] == m) {
            Xn->grad[offset] += share;
          }
          for (int d = int(R2) - 1; d >= 0; --d) {
            red_coord[d]++;
            offset += red_strides2[d];
            if (red_coord[d] < red_extents2[d]) break;
            offset -= red_strides2[d] * red_extents2[d];
            red_coord[d] = 0;
          }
        }
      }
    });
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
