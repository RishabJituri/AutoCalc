#include "ag/ops/elementwise.hpp"
#include "ag/ops/tensor_utils.hpp"
#include <cmath>
#include "ag/parallel/parallel_for.hpp"

namespace ag {
using detail::broadcast_two;
using detail::numel;
using detail::strides_for;
using detail::unravel_index;

// Parallelization params for elementwise ops
static constexpr std::size_t ELEM_SERIAL_CUTOFF = 4096;
static constexpr std::size_t ELEM_GRAIN = 1024;

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
  out->value.assign(oN, 0.0f);
  out->grad.assign(oN, 0.0f);
  out->requires_grad = (A.n->requires_grad || B.n->requires_grad);
  out->parents = {A.n, B.n};

  const auto As = strides_for(A.n->shape), Bs = strides_for(B.n->shape);

  // forward
  if (oN < ELEM_SERIAL_CUTOFF) {
    for (std::size_t lin = 0; lin < oN; ++lin) {
      auto idx = detail::unravel_index(lin, out_shape);
      const std::size_t ai = map_aligned(idx, out_shape, A.n->shape, As);
      const std::size_t bi = map_aligned(idx, out_shape, B.n->shape, Bs);
      out->value[lin] = static_cast<float>(A.n->value[ai]) + static_cast<float>(B.n->value[bi]);
    }
  } else {
    const auto outp = out->value.data();
    const auto aval = A.n->value.data();
    const auto bval = B.n->value.data();
    ag::parallel::parallel_for(oN, ELEM_GRAIN, [&](std::size_t l0, std::size_t l1){
      for (std::size_t lin = l0; lin < l1; ++lin) {
        auto idx = detail::unravel_index(lin, out_shape);
        const std::size_t ai = map_aligned(idx, out_shape, A.n->shape, As);
        const std::size_t bi = map_aligned(idx, out_shape, B.n->shape, Bs);
        outp[lin] = static_cast<float>(aval[ai]) + static_cast<float>(bval[bi]);
      }
    });
  }

  std::weak_ptr<Node> ow = out, aw = A.n, bw = B.n;
  out->backward = [ow, aw, bw, out_shape, As, Bs]() {
    auto o = ow.lock(); if (!o) return;
    auto a = aw.lock(); auto b = bw.lock();
    const std::size_t oN = o->value.size();

    ag::parallel::ScopedDeterministicParallel scope_guard;

    const bool A_nobroadcast = a && (a->shape == out_shape);
    const bool B_nobroadcast = b && (b->shape == out_shape);

    // If both inputs have the same shape as output we can parallelize a single pass safely.
    if (oN >= ELEM_SERIAL_CUTOFF && A_nobroadcast && B_nobroadcast) {
      const auto g_in = o->grad.data();
      auto a_out = a->grad.data();
      auto b_out = b->grad.data();
      ag::parallel::parallel_for(oN, ELEM_GRAIN, [&](std::size_t l0, std::size_t l1){
        for (std::size_t lin = l0; lin < l1; ++lin) {
          const float g = static_cast<float>(g_in[lin]);
          a_out[lin] += g;
          b_out[lin] += g;
        }
      });
      return;
    }

    // Otherwise, fall back to safe per-input accumulation (parallelize per-input only if no-broadcast)
    if (a && a->requires_grad) {
      if (oN >= ELEM_SERIAL_CUTOFF && A_nobroadcast) {
        const auto g_in = o->grad.data();
        auto a_out = a->grad.data();
        ag::parallel::parallel_for(oN, ELEM_GRAIN, [&](std::size_t l0, std::size_t l1){
          for (std::size_t lin = l0; lin < l1; ++lin) a_out[lin] += static_cast<float>(g_in[lin]);
        });
      } else {
        for (std::size_t lin = 0; lin < oN; ++lin) {
          const auto idx = detail::unravel_index(lin, out_shape);
          const std::size_t ai = map_aligned(idx, out_shape, a->shape, As);
          a->grad[ai] += static_cast<float>(o->grad[lin]);
        }
      }
    }

    if (b && b->requires_grad) {
      if (oN >= ELEM_SERIAL_CUTOFF && B_nobroadcast) {
        const auto g_in = o->grad.data();
        auto b_out = b->grad.data();
        ag::parallel::parallel_for(oN, ELEM_GRAIN, [&](std::size_t l0, std::size_t l1){
          for (std::size_t lin = l0; lin < l1; ++lin) b_out[lin] += static_cast<float>(g_in[lin]);
        });
      } else {
        for (std::size_t lin = 0; lin < oN; ++lin) {
          const auto idx = detail::unravel_index(lin, out_shape);
          const std::size_t bi = map_aligned(idx, out_shape, b->shape, Bs);
          b->grad[bi] += static_cast<float>(o->grad[lin]);
        }
      }
    }
  };

  return make_from_node(out);
}

Variable sub(const Variable& A, const Variable& B) {
  const auto out_shape = broadcast_two(A.n->shape, B.n->shape);
  const std::size_t oN = numel(out_shape);

  auto out = std::make_shared<Node>();
  out->shape = out_shape; out->value.assign(oN, 0.0f); out->grad.assign(oN, 0.0f);
  out->requires_grad = (A.n->requires_grad || B.n->requires_grad);
  out->parents = {A.n, B.n};

  const auto As = strides_for(A.n->shape), Bs = strides_for(B.n->shape);

  if (oN < ELEM_SERIAL_CUTOFF) {
    for (std::size_t lin = 0; lin < oN; ++lin) {
      const auto idx = detail::unravel_index(lin, out_shape);
      const std::size_t ai = map_aligned(idx, out_shape, A.n->shape, As);
      const std::size_t bi = map_aligned(idx, out_shape, B.n->shape, Bs);
      out->value[lin] = static_cast<float>(A.n->value[ai]) - static_cast<float>(B.n->value[bi]);
    }
  } else {
    const auto outp = out->value.data();
    const auto aval = A.n->value.data();
    const auto bval = B.n->value.data();
    ag::parallel::parallel_for(oN, ELEM_GRAIN, [&](std::size_t l0, std::size_t l1){
      for (std::size_t lin = l0; lin < l1; ++lin) {
        const auto idx = detail::unravel_index(lin, out_shape);
        const std::size_t ai = map_aligned(idx, out_shape, A.n->shape, As);
        const std::size_t bi = map_aligned(idx, out_shape, B.n->shape, Bs);
        outp[lin] = static_cast<float>(aval[ai]) - static_cast<float>(bval[bi]);
      }
    });
  }

  std::weak_ptr<Node> ow = out, aw = A.n, bw = B.n;
  out->backward = [ow, aw, bw, out_shape, As, Bs]() {
    auto o = ow.lock(); if (!o) return; auto a = aw.lock(); auto b = bw.lock();
    const std::size_t oN = o->value.size();
    ag::parallel::ScopedDeterministicParallel scope_guard;
    const bool A_nobroadcast = a && (a->shape == out_shape);
    const bool B_nobroadcast = b && (b->shape == out_shape);

    if (oN >= ELEM_SERIAL_CUTOFF && A_nobroadcast && B_nobroadcast) {
      const auto g_in = o->grad.data();
      auto a_out = a->grad.data();
      auto b_out = b->grad.data();
      ag::parallel::parallel_for(oN, ELEM_GRAIN, [&](std::size_t l0, std::size_t l1){
        for (std::size_t lin = l0; lin < l1; ++lin) {
          const float g = static_cast<float>(g_in[lin]);
          a_out[lin] += g;
          b_out[lin] -= g;
        }
      });
      return;
    }

    if (a && a->requires_grad) {
      if (oN >= ELEM_SERIAL_CUTOFF && A_nobroadcast) {
        const auto g_in = o->grad.data(); auto a_out = a->grad.data();
        ag::parallel::parallel_for(oN, ELEM_GRAIN, [&](std::size_t l0, std::size_t l1){
          for (std::size_t lin = l0; lin < l1; ++lin) a_out[lin] += static_cast<float>(g_in[lin]);
        });
      } else {
        for (std::size_t lin = 0; lin < oN; ++lin) {
          const auto idx = detail::unravel_index(lin, out_shape);
          const std::size_t ai = map_aligned(idx, out_shape, a->shape, As);
          a->grad[ai] += static_cast<float>(o->grad[lin]);
        }
      }
    }

    if (b && b->requires_grad) {
      if (oN >= ELEM_SERIAL_CUTOFF && B_nobroadcast) {
        const auto g_in = o->grad.data(); auto b_out = b->grad.data();
        ag::parallel::parallel_for(oN, ELEM_GRAIN, [&](std::size_t l0, std::size_t l1){
          for (std::size_t lin = l0; lin < l1; ++lin) b_out[lin] += -static_cast<float>(g_in[lin]);
        });
      } else {
        for (std::size_t lin = 0; lin < oN; ++lin) {
          const auto idx = detail::unravel_index(lin, out_shape);
          const std::size_t bi = map_aligned(idx, out_shape, b->shape, Bs);
          b->grad[bi] -= static_cast<float>(o->grad[lin]);
        }
      }
    }
  };

  return make_from_node(out);
}

Variable mul(const Variable& A, const Variable& B) {
  const auto out_shape = broadcast_two(A.n->shape, B.n->shape);
  const std::size_t oN = numel(out_shape);

  auto out = std::make_shared<Node>();
  out->shape = out_shape; out->value.assign(oN, 0.0f); out->grad.assign(oN, 0.0f);
  out->requires_grad = (A.n->requires_grad || B.n->requires_grad);
  out->parents = {A.n, B.n};

  const auto As = strides_for(A.n->shape), Bs = strides_for(B.n->shape);

  if (oN < ELEM_SERIAL_CUTOFF) {
    for (std::size_t lin = 0; lin < oN; ++lin) {
      const auto idx = detail::unravel_index(lin, out_shape);
      const std::size_t ai = map_aligned(idx, out_shape, A.n->shape, As);
      const std::size_t bi = map_aligned(idx, out_shape, B.n->shape, Bs);
      out->value[lin] = static_cast<float>(A.n->value[ai]) * static_cast<float>(B.n->value[bi]);
    }
  } else {
    const auto outp = out->value.data();
    const auto aval = A.n->value.data();
    const auto bval = B.n->value.data();
    ag::parallel::parallel_for(oN, ELEM_GRAIN, [&](std::size_t l0, std::size_t l1){
      for (std::size_t lin = l0; lin < l1; ++lin) {
        const auto idx = detail::unravel_index(lin, out_shape);
        const std::size_t ai = map_aligned(idx, out_shape, A.n->shape, As);
        const std::size_t bi = map_aligned(idx, out_shape, B.n->shape, Bs);
        outp[lin] = static_cast<float>(aval[ai]) * static_cast<float>(bval[bi]);
      }
    });
  }

  std::weak_ptr<Node> ow = out, aw = A.n, bw = B.n;
  out->backward = [ow, aw, bw, out_shape, As, Bs]() {
    auto o = ow.lock(); if (!o) return; auto a = aw.lock(); auto b = bw.lock();
    const std::size_t oN = o->value.size();
    ag::parallel::ScopedDeterministicParallel scope_guard;
    const bool A_nobroadcast = a && (a->shape == out_shape);
    const bool B_nobroadcast = b && (b->shape == out_shape);

    if (oN >= ELEM_SERIAL_CUTOFF && A_nobroadcast && B_nobroadcast) {
      const auto g_in = o->grad.data();
      const auto aval = a->value.data();
      const auto bval = b->value.data();
      auto a_out = a->grad.data();
      auto b_out = b->grad.data();
      ag::parallel::parallel_for(oN, ELEM_GRAIN, [&](std::size_t l0, std::size_t l1){
        for (std::size_t lin = l0; lin < l1; ++lin) {
          const float g = static_cast<float>(g_in[lin]);
          const float bv = static_cast<float>(bval[lin]);
          const float av = static_cast<float>(aval[lin]);
          a_out[lin] += g * bv;
          b_out[lin] += g * av;
        }
      });
      return;
    }

    if (a && a->requires_grad) {
      if (oN >= ELEM_SERIAL_CUTOFF && A_nobroadcast) {
        const auto g_in = o->grad.data(); const auto bval = b->value.data(); auto a_out = a->grad.data();
        ag::parallel::parallel_for(oN, ELEM_GRAIN, [&](std::size_t l0, std::size_t l1){
          for (std::size_t lin = l0; lin < l1; ++lin) a_out[lin] += static_cast<float>(g_in[lin]) * static_cast<float>(bval[lin]);
        });
      } else {
        for (std::size_t lin = 0; lin < oN; ++lin) {
          const auto idx = detail::unravel_index(lin, out_shape);
          const std::size_t ai = map_aligned(idx, out_shape, a->shape, As);
          const std::size_t bi = map_aligned(idx, out_shape, b->shape, Bs);
          a->grad[ai] += static_cast<float>(o->grad[lin]) * (b ? static_cast<float>(b->value[bi]) : 0.0f);
        }
      }
    }

    if (b && b->requires_grad) {
      if (oN >= ELEM_SERIAL_CUTOFF && B_nobroadcast) {
        const auto g_in = o->grad.data(); const auto aval = a->value.data(); auto b_out = b->grad.data();
        ag::parallel::parallel_for(oN, ELEM_GRAIN, [&](std::size_t l0, std::size_t l1){
          for (std::size_t lin = l0; lin < l1; ++lin) b_out[lin] += static_cast<float>(g_in[lin]) * static_cast<float>(aval[lin]);
        });
      } else {
        for (std::size_t lin = 0; lin < oN; ++lin) {
          const auto idx = detail::unravel_index(lin, out_shape);
          const std::size_t ai = map_aligned(idx, out_shape, a->shape, As);
          const std::size_t bi = map_aligned(idx, out_shape, b->shape, Bs);
          b->grad[bi] += static_cast<float>(o->grad[lin]) * (a ? static_cast<float>(a->value[ai]) : 0.0f);
        }
      }
    }
  };

  return make_from_node(out);
}

Variable div(const Variable& A, const Variable& B) {
  const auto out_shape = broadcast_two(A.n->shape, B.n->shape);
  const std::size_t oN = numel(out_shape);

  auto out = std::make_shared<Node>();
  out->shape = out_shape; out->value.assign(oN, 0.0f); out->grad.assign(oN, 0.0f);
  out->requires_grad = (A.n->requires_grad || B.n->requires_grad);
  out->parents = {A.n, B.n};

  const auto As = strides_for(A.n->shape), Bs = strides_for(B.n->shape);

  if (oN < ELEM_SERIAL_CUTOFF) {
    for (std::size_t lin = 0; lin < oN; ++lin) {
      const auto idx = detail::unravel_index(lin, out_shape);
      const std::size_t ai = map_aligned(idx, out_shape, A.n->shape, As);
      const std::size_t bi = map_aligned(idx, out_shape, B.n->shape, Bs);
      out->value[lin] = static_cast<float>(A.n->value[ai]) / static_cast<float>(B.n->value[bi]);
    }
  } else {
    const auto outp = out->value.data();
    const auto aval = A.n->value.data();
    const auto bval = B.n->value.data();
    ag::parallel::parallel_for(oN, ELEM_GRAIN, [&](std::size_t l0, std::size_t l1){
      for (std::size_t lin = l0; lin < l1; ++lin) {
        const auto idx = detail::unravel_index(lin, out_shape);
        const std::size_t ai = map_aligned(idx, out_shape, A.n->shape, As);
        const std::size_t bi = map_aligned(idx, out_shape, B.n->shape, Bs);
        outp[lin] = static_cast<float>(aval[ai]) / static_cast<float>(bval[bi]);
      }
    });
  }

  std::weak_ptr<Node> ow = out, aw = A.n, bw = B.n;
  out->backward = [ow, aw, bw, out_shape, As, Bs]() {
    auto o = ow.lock(); if (!o) return; auto a = aw.lock(); auto b = bw.lock();
    const std::size_t oN = o->value.size();
    ag::parallel::ScopedDeterministicParallel scope_guard;
    const bool A_nobroadcast = a && (a->shape == out_shape);
    const bool B_nobroadcast = b && (b->shape == out_shape);

    if (oN >= ELEM_SERIAL_CUTOFF && A_nobroadcast && B_nobroadcast) {
      const auto g_in = o->grad.data(); const auto aval = a->value.data(); const auto bval = b->value.data();
      auto a_out = a->grad.data(); auto b_out = b->grad.data();
      ag::parallel::parallel_for(oN, ELEM_GRAIN, [&](std::size_t l0, std::size_t l1){
        for (std::size_t lin = l0; lin < l1; ++lin) {
          const float g = static_cast<float>(g_in[lin]);
          const float bv = static_cast<float>(bval[lin]);
          const float av = static_cast<float>(aval[lin]);
          a_out[lin] += g / bv;
          b_out[lin] += -g * av / (bv * bv);
        }
      });
      return;
    }

    if (a && a->requires_grad) {
      if (oN >= ELEM_SERIAL_CUTOFF && A_nobroadcast) {
        const auto g_in = o->grad.data(); const auto bval = b->value.data(); auto a_out = a->grad.data();
        ag::parallel::parallel_for(oN, ELEM_GRAIN, [&](std::size_t l0, std::size_t l1){
          for (std::size_t lin = l0; lin < l1; ++lin) a_out[lin] += static_cast<float>(g_in[lin]) / static_cast<float>(bval[lin]);
        });
      } else {
        for (std::size_t lin = 0; lin < oN; ++lin) {
          const auto idx = detail::unravel_index(lin, out_shape);
          const std::size_t ai = map_aligned(idx, out_shape, a->shape, As);
          const std::size_t bi = map_aligned(idx, out_shape, b->shape, Bs);
          const float denom = (b ? static_cast<float>(b->value[bi]) : 1.0f);
          a->grad[ai] += static_cast<float>(o->grad[lin]) / denom;
        }
      }
    }

    if (b && b->requires_grad) {
      if (oN >= ELEM_SERIAL_CUTOFF && B_nobroadcast) {
        const auto g_in = o->grad.data(); const auto aval = a->value.data(); auto b_out = b->grad.data();
        ag::parallel::parallel_for(oN, ELEM_GRAIN, [&](std::size_t l0, std::size_t l1){
          for (std::size_t lin = l0; lin < l1; ++lin) b_out[lin] += - static_cast<float>(g_in[lin]) * static_cast<float>(aval[lin]) / (static_cast<float>(b->value[lin]) * static_cast<float>(b->value[lin]));
        });
      } else {
        for (std::size_t lin = 0; lin < oN; ++lin) {
          const auto idx = detail::unravel_index(lin, out_shape);
          const std::size_t ai = map_aligned(idx, out_shape, a->shape, As);
          const std::size_t bi = map_aligned(idx, out_shape, b->shape, Bs);
          const float denom = (b ? static_cast<float>(b->value[bi]) : 1.0f);
          b->grad[bi] += - static_cast<float>(o->grad[lin]) * (a ? static_cast<float>(a->value[ai]) : 0.0f) / (denom*denom);
        }
      }
    }
  };

  return make_from_node(out);
}

Variable neg(const Variable& X) {
  auto out = std::make_shared<Node>();
  out->shape = X.n->shape; out->value = X.n->value; out->grad.assign(out->value.size(), 0.0f);
  for (float& v : out->value) v = -v;
  out->requires_grad = X.n->requires_grad; out->parents = {X.n};

  std::weak_ptr<Node> ow = out, xw = X.n;
  out->backward = [ow, xw]() {
    auto o = ow.lock(); auto x = xw.lock(); if (!o || !x || !x->requires_grad) return;
    for (std::size_t i = 0; i < o->grad.size(); ++i) x->grad[i] += -static_cast<float>(o->grad[i]);
  };
  return make_from_node(out);
}

Variable sinv(const Variable& X) {
  auto out = std::make_shared<Node>();
  out->shape = X.n->shape; out->value.assign(X.n->value.size(), 0.0f); out->grad.assign(X.n->value.size(), 0.0f);
  out->requires_grad = X.n->requires_grad; out->parents = {X.n};
  for (std::size_t i=0;i<X.n->value.size();++i) out->value[i] = std::sin(static_cast<float>(X.n->value[i]));

  std::weak_ptr<Node> ow = out, xw = X.n;
  out->backward = [ow, xw]() {
    auto o = ow.lock(); auto x = xw.lock(); if (!o || !x || !x->requires_grad) return;
    for (std::size_t i=0;i<o->grad.size();++i) x->grad[i] += static_cast<float>(o->grad[i]) * std::cos(static_cast<float>(x->value[i]));
  };
  return make_from_node(out);
}

Variable cosv(const Variable& X) {
  auto out = std::make_shared<Node>();
  out->shape = X.n->shape; out->value.assign(X.n->value.size(), 0.0f); out->grad.assign(X.n->value.size(), 0.0f);
  out->requires_grad = X.n->requires_grad; out->parents = {X.n};
  for (std::size_t i=0;i<X.n->value.size();++i) out->value[i] = std::cos(static_cast<float>(X.n->value[i]));

  std::weak_ptr<Node> ow = out, xw = X.n;
  out->backward = [ow, xw]() {
    auto o = ow.lock(); auto x = xw.lock(); if (!o || !x || !x->requires_grad) return;
    for (std::size_t i=0;i<o->grad.size();++i) x->grad[i] += -static_cast<float>(o->grad[i]) * std::sin(static_cast<float>(x->value[i]));
  };
  return make_from_node(out);
}

Variable expv(const Variable& X) {
  auto out = std::make_shared<Node>();
  out->shape = X.n->shape; out->value.assign(X.n->value.size(), 0.0f); out->grad.assign(X.n->value.size(), 0.0f);
  out->requires_grad = X.n->requires_grad; out->parents = {X.n};
  for (std::size_t i=0;i<X.n->value.size();++i) out->value[i] = std::exp(static_cast<float>(X.n->value[i]));

  std::weak_ptr<Node> ow = out, xw = X.n;
  out->backward = [ow, xw]() {
    auto o = ow.lock(); auto x = xw.lock(); if (!o || !x || !x->requires_grad) return;
    for (std::size_t i=0;i<o->grad.size();++i) x->grad[i] += static_cast<float>(o->grad[i]) * static_cast<float>(o->value[i]); // d/dx exp(x) = exp(x)
  };
  return make_from_node(out);
}

Variable pow(const Variable& A, const Variable& P) {
  const auto out_shape = detail::broadcast_two(A.n->shape, P.n->shape);
  const std::size_t oN = numel(out_shape);

  auto out = std::make_shared<Node>();
  out->shape = out_shape; out->value.assign(oN, 0.0f); out->grad.assign(oN, 0.0f);
  out->requires_grad = (A.n->requires_grad || P.n->requires_grad);
  out->parents = {A.n, P.n};

  const auto As = strides_for(A.n->shape), Ps = strides_for(P.n->shape);

  for (std::size_t lin = 0; lin < oN; ++lin) {
    const auto idx = detail::unravel_index(lin, out_shape);
    const std::size_t ai = map_aligned(idx, out_shape, A.n->shape, As);
    const std::size_t pi = map_aligned(idx, out_shape, P.n->shape, Ps);
    out->value[lin] = std::pow(static_cast<float>(A.n->value[ai]), static_cast<float>(P.n->value[pi]));
  }

  std::weak_ptr<Node> ow = out, aw = A.n, pw = P.n;
  out->backward = [ow, aw, pw, out_shape, As, Ps]() {
    auto o = ow.lock(); if (!o) return; auto a = aw.lock(); auto p = pw.lock();
    for (std::size_t lin = 0; lin < o->value.size(); ++lin) {
      const auto idx = detail::unravel_index(lin, out_shape);
      const std::size_t ai = a ? map_aligned(idx, out_shape, a->shape, As) : 0;
      const std::size_t pi = p ? map_aligned(idx, out_shape, p->shape, Ps) : 0;

      if (a && a->requires_grad) {
        const float y = static_cast<float>(a->value[ai]);
        const float pe = p ? static_cast<float>(p->value[pi]) : 0.0f;
        if (!(y == 0.0f && pe < 1.0f)) a->grad[ai] += static_cast<float>(o->grad[lin]) * (pe * std::pow(y, pe - 1.0f));
      }
      if (p && p->requires_grad) {
        const float y = a ? static_cast<float>(a->value[ai]) : 1.0f;
        const float ln_y = (y > 0.0f) ? std::log(y) : 0.0f;
        p->grad[pi] += static_cast<float>(o->grad[lin]) * (static_cast<float>(o->value[lin]) * ln_y);
      }
    }
  };

  return make_from_node(out);
}

} // namespace ag
