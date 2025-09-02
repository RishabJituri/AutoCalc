#include "ag/ops/elementwise.hpp"
#include "ag/ops/tensor_utils.hpp"
#include <cmath>

namespace ag {
using detail::broadcast_two;
using detail::numel;
using detail::strides_for;
using detail::unravel_index;

// Map an output index to the flat index in input A (with broadcasting)
static std::size_t map_aligned(const std::vector<std::size_t>& out_idx,
                               const std::vector<std::size_t>& out_shape,
                               const std::vector<std::size_t>& ashape,
                               const std::vector<std::size_t>& astrides)
{
  const std::size_t r = out_shape.size();
  const std::size_t ra = ashape.size();
  std::size_t off = 0;
  for (std::size_t d = 0; d < r; ++d) {
    if (d < r - ra) continue; // dims not present in A
    const std::size_t ai = d - (r - ra);
    const bool broadcast = (ashape[ai] == 1);
    const std::size_t coord = broadcast ? 0 : out_idx[d];
    off += coord * astrides[ai];
  }
  return off;
}

Variable add(const Variable& A, const Variable& B) {
  const auto out_shape = broadcast_two(A.n->shape, B.n->shape);
  const std::size_t oN = numel(out_shape);

  auto out = std::make_shared<Node>();
  out->shape = out_shape;
  out->value.assign(oN, 0.0);
  out->grad.assign(oN, 0.0);
  out->requires_grad = (A.n->requires_grad || B.n->requires_grad);
  out->parents = {A.n, B.n};

  const auto As = strides_for(A.n->shape), Bs = strides_for(B.n->shape);

  // forward
  for (std::size_t lin = 0; lin < oN; ++lin) {
    auto idx = detail::unravel_index(lin, out_shape);
    const std::size_t ai = map_aligned(idx, out_shape, A.n->shape, As);
    const std::size_t bi = map_aligned(idx, out_shape, B.n->shape, Bs);
    out->value[lin] = A.n->value[ai] + B.n->value[bi];
  }

  std::weak_ptr<Node> ow = out, aw = A.n, bw = B.n;
  out->backward = [ow, aw, bw, out_shape, As, Bs]() {
    auto o = ow.lock(); if (!o) return;
    auto a = aw.lock(); auto b = bw.lock();
    const std::size_t oN = o->value.size();
    for (std::size_t lin = 0; lin < oN; ++lin) {
      auto idx = detail::unravel_index(lin, out_shape);
      if (a && a->requires_grad) {
        const std::size_t ai = map_aligned(idx, out_shape, a->shape, As);
        a->grad[ai] += o->grad[lin];            // dy/da = 1
      }
      if (b && b->requires_grad) {
        const std::size_t bi = map_aligned(idx, out_shape, b->shape, Bs);
        b->grad[bi] += o->grad[lin];            // dy/dc = 1
      }
    }
  };

  return make_from_node(out);
}

Variable sub(const Variable& A, const Variable& B) {
  const auto out_shape = broadcast_two(A.n->shape, B.n->shape);
  const std::size_t oN = numel(out_shape);

  auto out = std::make_shared<Node>();
  out->shape = out_shape; out->value.assign(oN, 0.0); out->grad.assign(oN, 0.0);
  out->requires_grad = (A.n->requires_grad || B.n->requires_grad);
  out->parents = {A.n, B.n};

  const auto As = strides_for(A.n->shape), Bs = strides_for(B.n->shape);

  for (std::size_t lin = 0; lin < oN; ++lin) {
    const auto idx = detail::unravel_index(lin, out_shape);
    const std::size_t ai = map_aligned(idx, out_shape, A.n->shape, As);
    const std::size_t bi = map_aligned(idx, out_shape, B.n->shape, Bs);
    out->value[lin] = A.n->value[ai] - B.n->value[bi];
  }

  std::weak_ptr<Node> ow = out, aw = A.n, bw = B.n;
  out->backward = [ow, aw, bw, out_shape, As, Bs]() {
    auto o = ow.lock(); if (!o) return; auto a = aw.lock(); auto b = bw.lock();
    for (std::size_t lin = 0; lin < o->value.size(); ++lin) {
      auto idx = detail::unravel_index(lin, out_shape);
      if (a && a->requires_grad) {
        const std::size_t ai = map_aligned(idx, out_shape, a->shape, As);
        a->grad[ai] += o->grad[lin];
      }
      if (b && b->requires_grad) {
        const std::size_t bi = map_aligned(idx, out_shape, b->shape, Bs);
        b->grad[bi] -= o->grad[lin];            // minus sign
      }
    }
  };

  return make_from_node(out);
}

Variable mul(const Variable& A, const Variable& B) {
  const auto out_shape = broadcast_two(A.n->shape, B.n->shape);
  const std::size_t oN = numel(out_shape);

  auto out = std::make_shared<Node>();
  out->shape = out_shape; out->value.assign(oN, 0.0); out->grad.assign(oN, 0.0);
  out->requires_grad = (A.n->requires_grad || B.n->requires_grad);
  out->parents = {A.n, B.n};

  const auto As = strides_for(A.n->shape), Bs = strides_for(B.n->shape);

  for (std::size_t lin = 0; lin < oN; ++lin) {
    const auto idx = detail::unravel_index(lin, out_shape);
    const std::size_t ai = map_aligned(idx, out_shape, A.n->shape, As);
    const std::size_t bi = map_aligned(idx, out_shape, B.n->shape, Bs);
    out->value[lin] = A.n->value[ai] * B.n->value[bi];
  }

  std::weak_ptr<Node> ow = out, aw = A.n, bw = B.n;
  out->backward = [ow, aw, bw, out_shape, As, Bs]() {
    auto o = ow.lock(); if (!o) return; auto a = aw.lock(); auto b = bw.lock();
    for (std::size_t lin = 0; lin < o->value.size(); ++lin) {
      const auto idx = detail::unravel_index(lin, out_shape);
      const std::size_t ai = a ? map_aligned(idx, out_shape, a->shape, As) : 0;
      const std::size_t bi = b ? map_aligned(idx, out_shape, b->shape, Bs) : 0;
      if (a && a->requires_grad) a->grad[ai] += o->grad[lin] * (b ? b->value[bi] : 0.0);
      if (b && b->requires_grad) b->grad[bi] += o->grad[lin] * (a ? a->value[ai] : 0.0);
    }
  };

  return make_from_node(out);
}

Variable div(const Variable& A, const Variable& B) {
  const auto out_shape = broadcast_two(A.n->shape, B.n->shape);
  const std::size_t oN = numel(out_shape);

  auto out = std::make_shared<Node>();
  out->shape = out_shape; out->value.assign(oN, 0.0); out->grad.assign(oN, 0.0);
  out->requires_grad = (A.n->requires_grad || B.n->requires_grad);
  out->parents = {A.n, B.n};

  const auto As = strides_for(A.n->shape), Bs = strides_for(B.n->shape);

  for (std::size_t lin = 0; lin < oN; ++lin) {
    const auto idx = detail::unravel_index(lin, out_shape);
    const std::size_t ai = map_aligned(idx, out_shape, A.n->shape, As);
    const std::size_t bi = map_aligned(idx, out_shape, B.n->shape, Bs);
    out->value[lin] = A.n->value[ai] / B.n->value[bi];
  }

  std::weak_ptr<Node> ow = out, aw = A.n, bw = B.n;
  out->backward = [ow, aw, bw, out_shape, As, Bs]() {
    auto o = ow.lock(); if (!o) return; auto a = aw.lock(); auto b = bw.lock();
    for (std::size_t lin = 0; lin < o->value.size(); ++lin) {
      const auto idx = detail::unravel_index(lin, out_shape);
      const std::size_t ai = a ? map_aligned(idx, out_shape, a->shape, As) : 0;
      const std::size_t bi = b ? map_aligned(idx, out_shape, b->shape, Bs) : 0;
      const double denom = (b ? b->value[bi] : 1.0);
      if (a && a->requires_grad) a->grad[ai] += o->grad[lin] / denom;
      if (b && b->requires_grad) b->grad[bi] += - o->grad[lin] * (a ? a->value[ai] : 0.0) / (denom*denom);
    }
  };

  return make_from_node(out);
}

Variable neg(const Variable& X) {
  auto out = std::make_shared<Node>();
  out->shape = X.n->shape; out->value = X.n->value; out->grad.assign(out->value.size(), 0.0);
  for (double& v : out->value) v = -v;
  out->requires_grad = X.n->requires_grad; out->parents = {X.n};

  std::weak_ptr<Node> ow = out, xw = X.n;
  out->backward = [ow, xw]() {
    auto o = ow.lock(); auto x = xw.lock(); if (!o || !x || !x->requires_grad) return;
    for (std::size_t i = 0; i < o->grad.size(); ++i) x->grad[i] += -o->grad[i];
  };
  return make_from_node(out);
}

Variable sinv(const Variable& X) {
  auto out = std::make_shared<Node>();
  out->shape = X.n->shape; out->value.assign(X.n->value.size(), 0.0); out->grad.assign(X.n->value.size(), 0.0);
  out->requires_grad = X.n->requires_grad; out->parents = {X.n};
  for (std::size_t i=0;i<X.n->value.size();++i) out->value[i] = std::sin(X.n->value[i]);

  std::weak_ptr<Node> ow = out, xw = X.n;
  out->backward = [ow, xw]() {
    auto o = ow.lock(); auto x = xw.lock(); if (!o || !x || !x->requires_grad) return;
    for (std::size_t i=0;i<o->grad.size();++i) x->grad[i] += o->grad[i] * std::cos(x->value[i]);
  };
  return make_from_node(out);
}

Variable cosv(const Variable& X) {
  auto out = std::make_shared<Node>();
  out->shape = X.n->shape; out->value.assign(X.n->value.size(), 0.0); out->grad.assign(X.n->value.size(), 0.0);
  out->requires_grad = X.n->requires_grad; out->parents = {X.n};
  for (std::size_t i=0;i<X.n->value.size();++i) out->value[i] = std::cos(X.n->value[i]);

  std::weak_ptr<Node> ow = out, xw = X.n;
  out->backward = [ow, xw]() {
    auto o = ow.lock(); auto x = xw.lock(); if (!o || !x || !x->requires_grad) return;
    for (std::size_t i=0;i<o->grad.size();++i) x->grad[i] += -o->grad[i] * std::sin(x->value[i]);
  };
  return make_from_node(out);
}

Variable expv(const Variable& X) {
  auto out = std::make_shared<Node>();
  out->shape = X.n->shape; out->value.assign(X.n->value.size(), 0.0); out->grad.assign(X.n->value.size(), 0.0);
  out->requires_grad = X.n->requires_grad; out->parents = {X.n};
  for (std::size_t i=0;i<X.n->value.size();++i) out->value[i] = std::exp(X.n->value[i]);

  std::weak_ptr<Node> ow = out, xw = X.n;
  out->backward = [ow, xw]() {
    auto o = ow.lock(); auto x = xw.lock(); if (!o || !x || !x->requires_grad) return;
    for (std::size_t i=0;i<o->grad.size();++i) x->grad[i] += o->grad[i] * o->value[i]; // d/dx exp(x) = exp(x)
  };
  return make_from_node(out);
}

Variable pow(const Variable& A, const Variable& P) {
  const auto out_shape = detail::broadcast_two(A.n->shape, P.n->shape);
  const std::size_t oN = numel(out_shape);

  auto out = std::make_shared<Node>();
  out->shape = out_shape; out->value.assign(oN, 0.0); out->grad.assign(oN, 0.0);
  out->requires_grad = (A.n->requires_grad || P.n->requires_grad);
  out->parents = {A.n, P.n};

  const auto As = strides_for(A.n->shape), Ps = strides_for(P.n->shape);

  for (std::size_t lin = 0; lin < oN; ++lin) {
    const auto idx = detail::unravel_index(lin, out_shape);
    const std::size_t ai = map_aligned(idx, out_shape, A.n->shape, As);
    const std::size_t pi = map_aligned(idx, out_shape, P.n->shape, Ps);
    out->value[lin] = std::pow(A.n->value[ai], P.n->value[pi]);
  }

  std::weak_ptr<Node> ow = out, aw = A.n, pw = P.n;
  out->backward = [ow, aw, pw, out_shape, As, Ps]() {
    auto o = ow.lock(); if (!o) return; auto a = aw.lock(); auto p = pw.lock();
    for (std::size_t lin = 0; lin < o->value.size(); ++lin) {
      const auto idx = detail::unravel_index(lin, out_shape);
      const std::size_t ai = a ? map_aligned(idx, out_shape, a->shape, As) : 0;
      const std::size_t pi = p ? map_aligned(idx, out_shape, p->shape, Ps) : 0;

      if (a && a->requires_grad) {
        const double y = a->value[ai];
        const double pe = p ? p->value[pi] : 0.0;
        if (!(y == 0.0 && pe < 1.0)) a->grad[ai] += o->grad[lin] * (pe * std::pow(y, pe - 1.0));
      }
      if (p && p->requires_grad) {
        const double y = a ? a->value[ai] : 1.0;
        const double ln_y = (y > 0.0) ? std::log(y) : 0.0;
        p->grad[pi] += o->grad[lin] * (o->value[lin] * ln_y);
      }
    }
  };

  return make_from_node(out);
}

} // namespace ag
