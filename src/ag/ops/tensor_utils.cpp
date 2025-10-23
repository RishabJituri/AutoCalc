#include "ag/ops/tensor_utils.hpp"
#include "ag/core/variables.hpp"
#include "ag/parallel/parallel_for.hpp"
#include <stdexcept>
#include <algorithm>

namespace ag {
using detail::strides_for;
using detail::unravel_index;
using detail::ravel_index;
using detail::numel;

static std::size_t norm_index(long long i, std::size_t dim) {
  long long ii = i;
  if (ii < 0) ii += (long long)dim;
  if (ii < 0 || (std::size_t)ii >= dim) throw std::out_of_range("slice index out of range");
  return (std::size_t)ii;
}

Variable at(const Variable& X, const std::vector<Slice>& spec) {
  const auto& xs = X.n->shape;
  const std::size_t R = xs.size();
  if (spec.size() != R) throw std::invalid_argument("at(): spec rank mismatch");

  // Build output shape
  std::vector<std::size_t> out_shape; out_shape.reserve(R);
  std::vector<std::size_t> starts(R), steps(R), sizes(R);
  for (std::size_t d = 0; d < R; ++d) {
    const auto& s = spec[d];
    const std::size_t dim = xs[d];
    std::size_t st=0, sp=1, sz=0;
    if (s.is_index) {
      st = norm_index(s.start, dim);
      sp = 1; sz = 1;
    } else {
      std::size_t a = (std::size_t)std::max(0LL, s.start);
      std::size_t b = (s.stop <= 0) ? dim : (std::size_t)std::min<long long>(s.stop, (long long)dim);
      if (a > b) a = b;
      sp = (std::size_t)(s.step == 0 ? 1 : s.step);
      sz = (b > a) ? ((b - a + sp - 1) / sp) : 0;
      st = a;
    }
    starts[d] = st; steps[d] = sp; sizes[d] = sz;
    out_shape.push_back(sz);
  }

  auto out = std::make_shared<Node>();
  out->shape = out_shape;
  const std::size_t N = numel(out_shape);
  out->value.assign(N, 0.0f);
  out->grad.assign(N, 0.0f);
  out->requires_grad = X.n->requires_grad;
  out->parents = {X.n};

  const auto Xstr = strides_for(xs);
  const auto Ostr = strides_for(out_shape);

  // Forward gather
  const auto Xin = X.n->value.data();
  auto O = out->value.data();
  const std::size_t SERIAL_CUTOFF = 4096, GRAIN = 1024;
  if (N < SERIAL_CUTOFF) {
    for (std::size_t lin = 0; lin < N; ++lin) {
      auto idx = unravel_index(lin, out_shape);
      std::vector<std::size_t> src_idx(R);
      for (std::size_t d = 0; d < R; ++d) src_idx[d] = starts[d] + idx[d] * steps[d];
      const std::size_t xi = ravel_index(src_idx, Xstr);
      O[lin] = Xin[xi];
    }
  } else {
    ag::parallel::parallel_for(N, GRAIN, [&](std::size_t i0, std::size_t i1){
      std::vector<std::size_t> src_idx(R);
      for (std::size_t lin = i0; lin < i1; ++lin) {
        auto idx = unravel_index(lin, out_shape);
        for (std::size_t d = 0; d < R; ++d) src_idx[d] = starts[d] + idx[d] * steps[d];
        const std::size_t xi = ravel_index(src_idx, Xstr);
        O[lin] = Xin[xi];
      }
    });
  }

  std::weak_ptr<Node> ow = out, xw = X.n;
  out->backward = [ow, xw, xs, out_shape, starts, steps]() {
    auto o = ow.lock(); auto x = xw.lock(); if (!o || !x || !x->requires_grad) return;
    if (x->grad.size() != x->value.size()) x->grad.assign(x->value.size(), 0.0f);
    const auto Xstr = strides_for(xs);

    const std::size_t N = o->grad.size();
    const std::size_t SERIAL_CUTOFF = 4096, GRAIN = 1024;
    if (N < SERIAL_CUTOFF) {
      for (std::size_t lin = 0; lin < N; ++lin) {
        auto idx = unravel_index(lin, out_shape);
        std::vector<std::size_t> src_idx(xs.size());
        for (std::size_t d = 0; d < xs.size(); ++d) src_idx[d] = starts[d] + idx[d] * steps[d];
        const std::size_t xi = ravel_index(src_idx, Xstr);
        x->grad[xi] += o->grad[lin];
      }
    } else {
      ag::parallel::parallel_for(N, GRAIN, [&](std::size_t i0, std::size_t i1){
        std::vector<std::size_t> src_idx(xs.size());
        for (std::size_t lin = i0; lin < i1; ++lin) {
          auto idx = unravel_index(lin, out_shape);
          for (std::size_t d = 0; d < xs.size(); ++d) src_idx[d] = starts[d] + idx[d] * steps[d];
          const std::size_t xi = ravel_index(src_idx, Xstr);
          x->grad[xi] += o->grad[lin];
        }
      });
    }
  };

  return make_from_node(out);
}

} // namespace ag