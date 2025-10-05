#include "ag/ops/activations.hpp"
#include "ag/ops/tensor_utils.hpp"
#include <cmath>
#include <memory>   // <-- important for weak_ptr
#include "ag/parallel/parallel_for.hpp"

namespace ag {
using detail::numel;

// small-shape cutoff and grain for elementwise ops
static constexpr std::size_t EW_SERIAL_CUTOFF = 4096;
static constexpr std::size_t EW_GRAIN = 1024;

Variable relu(const Variable& X) {
  auto out = std::make_shared<Node>();
  out->shape = X.n->shape;
  out->value.resize(numel(out->shape));
  out->grad.resize(numel(out->shape), 0.0f);
  out->requires_grad = X.n->requires_grad;
  out->parents = {X.n};

  const std::size_t N = out->value.size();
  if (N < EW_SERIAL_CUTOFF) {
    for (std::size_t i = 0; i < N; ++i) {
      float v = static_cast<float>(X.n->value[i]);
      out->value[i] = v > 0.0f ? v : 0.0f;
    }
  } else {
    const auto xin = X.n->value.data();
    auto outp = out->value.data();
    ag::parallel::parallel_for(N, EW_GRAIN, [&](std::size_t i0, std::size_t i1){
      for (std::size_t i = i0; i < i1; ++i) {
        float v = static_cast<float>(xin[i]);
        outp[i] = v > 0.0f ? v : 0.0f;
      }
    });
  }

  out->backward = [Xn = X.n, oweak = std::weak_ptr<Node>(out)]() {
    auto op = oweak.lock(); if (!op) return;
    Node* o = op.get();
    if (!Xn || !Xn->requires_grad) return;

    // ensure grad buffer
    if (Xn->grad.size() != Xn->value.size()) Xn->grad.assign(Xn->value.size(), 0.0f);

    const std::size_t N = o->value.size();
    ag::parallel::ScopedDeterministicParallel scope_guard;
    if (N < EW_SERIAL_CUTOFF) {
      for (std::size_t i = 0; i < N; ++i) {
        float v = static_cast<float>(Xn->value[i]);
        Xn->grad[i] += (v > 0.0f ? 1.0f : 0.0f) * static_cast<float>(o->grad[i]);
      }
    } else {
      const auto xin = Xn->value.data();
      const auto g_in = o->grad.data();
      auto g_out = Xn->grad.data();
      ag::parallel::parallel_for(N, EW_GRAIN, [&](std::size_t i0, std::size_t i1){
        for (std::size_t i = i0; i < i1; ++i) {
          float v = static_cast<float>(xin[i]);
          g_out[i] += (v > 0.0f ? 1.0f : 0.0f) * static_cast<float>(g_in[i]);
        }
      });
    }
  };
  return make_from_node(out);
}

Variable sigmoid(const Variable& X) {
  auto out = std::make_shared<Node>();
  out->shape = X.n->shape;
  out->value.resize(numel(out->shape));
  out->grad.resize(numel(out->shape), 0.0f);
  out->requires_grad = X.n->requires_grad;
  out->parents = {X.n};

  const std::size_t N = out->value.size();
  if (N < EW_SERIAL_CUTOFF) {
    for (std::size_t i = 0; i < N; ++i) {
      out->value[i] = 1.0f / (1.0f + std::exp(-static_cast<float>(X.n->value[i])));
    }
  } else {
    const auto xin = X.n->value.data();
    auto outp = out->value.data();
    ag::parallel::parallel_for(N, EW_GRAIN, [&](std::size_t i0, std::size_t i1){
      for (std::size_t i = i0; i < i1; ++i) outp[i] = 1.0f / (1.0f + std::exp(-static_cast<float>(xin[i])));
    });
  }

  out->backward = [Xn = X.n, oweak = std::weak_ptr<Node>(out)]() {
    auto op = oweak.lock(); if (!op) return;
    Node* o = op.get();
    if (!Xn || !Xn->requires_grad) return;
    if (Xn->grad.size() != Xn->value.size()) Xn->grad.assign(Xn->value.size(), 0.0f);

    const std::size_t N = o->value.size();
    ag::parallel::ScopedDeterministicParallel scope_guard;
    if (N < EW_SERIAL_CUTOFF) {
      for (std::size_t i = 0; i < N; ++i) {
        float y = static_cast<float>(o->value[i]);
        Xn->grad[i] += (y * (1.0f - y)) * static_cast<float>(o->grad[i]);
      }
    } else {
      const auto outv = o->value.data();
      const auto g_in = o->grad.data();
      auto g_out = Xn->grad.data();
      ag::parallel::parallel_for(N, EW_GRAIN, [&](std::size_t i0, std::size_t i1){
        for (std::size_t i = i0; i < i1; ++i) g_out[i] += (static_cast<float>(outv[i]) * (1.0f - static_cast<float>(outv[i]))) * static_cast<float>(g_in[i]);
      });
    }
  };
  return make_from_node(out);
}

Variable tanhv(const Variable& X) {
  auto out = std::make_shared<Node>();
  out->shape = X.n->shape;
  out->value.resize(numel(out->shape));
  out->grad.resize(numel(out->shape), 0.0f);
  out->requires_grad = X.n->requires_grad;
  out->parents = {X.n};

  const std::size_t N = out->value.size();
  if (N < EW_SERIAL_CUTOFF) {
    for (std::size_t i = 0; i < N; ++i) out->value[i] = std::tanh(static_cast<float>(X.n->value[i]));
  } else {
    const auto xin = X.n->value.data();
    auto outp = out->value.data();
    ag::parallel::parallel_for(N, EW_GRAIN, [&](std::size_t i0, std::size_t i1){
      for (std::size_t i = i0; i < i1; ++i) outp[i] = std::tanh(static_cast<float>(xin[i]));
    });
  }

  out->backward = [Xn = X.n, oweak = std::weak_ptr<Node>(out)]() {
    auto op = oweak.lock(); if (!op) return;
    Node* o = op.get();
    if (!Xn || !Xn->requires_grad) return;
    if (Xn->grad.size() != Xn->value.size()) Xn->grad.assign(Xn->value.size(), 0.0f);

    const std::size_t N = o->value.size();
    ag::parallel::ScopedDeterministicParallel scope_guard;
    if (N < EW_SERIAL_CUTOFF) {
      for (std::size_t i = 0; i < N; ++i) {
        float y = static_cast<float>(o->value[i]);
        Xn->grad[i] += (1.0f - y * y) * static_cast<float>(o->grad[i]);
      }
    } else {
      const auto outv = o->value.data();
      const auto g_in = o->grad.data();
      auto g_out = Xn->grad.data();
      ag::parallel::parallel_for(N, EW_GRAIN, [&](std::size_t i0, std::size_t i1){
        for (std::size_t i = i0; i < i1; ++i) g_out[i] += (1.0f - static_cast<float>(outv[i]) * static_cast<float>(outv[i])) * static_cast<float>(g_in[i]);
      });
    }
  };
  return make_from_node(out);
}

Variable logv(const Variable& X) {
  auto out = std::make_shared<Node>();
  out->shape = X.n->shape;
  out->value.resize(numel(out->shape));
  out->grad.resize(numel(out->shape), 0.0f);
  out->requires_grad = X.n->requires_grad;
  out->parents = {X.n};

  const std::size_t N = out->value.size();
  if (N < EW_SERIAL_CUTOFF) {
    for (std::size_t i = 0; i < N; ++i) out->value[i] = std::log(static_cast<float>(X.n->value[i]));
  } else {
    const auto xin = X.n->value.data();
    auto outp = out->value.data();
    ag::parallel::parallel_for(N, EW_GRAIN, [&](std::size_t i0, std::size_t i1){
      for (std::size_t i = i0; i < i1; ++i) outp[i] = std::log(static_cast<float>(xin[i]));
    });
  }

  out->backward = [Xn = X.n, oweak = std::weak_ptr<Node>(out)]() {
    auto op = oweak.lock(); if (!op) return;
    Node* o = op.get();
    if (!Xn || !Xn->requires_grad) return;
    if (Xn->grad.size() != Xn->value.size()) Xn->grad.assign(Xn->value.size(), 0.0f);

    const std::size_t N = o->value.size();
    ag::parallel::ScopedDeterministicParallel scope_guard;
    if (N < EW_SERIAL_CUTOFF) {
      for (std::size_t i = 0; i < N; ++i) {
        Xn->grad[i] += (1.0f / static_cast<float>(Xn->value[i])) * static_cast<float>(o->grad[i]);
      }
    } else {
      const auto g_in = o->grad.data();
      auto g_out = Xn->grad.data();
      const auto xin = Xn->value.data();
      ag::parallel::parallel_for(N, EW_GRAIN, [&](std::size_t i0, std::size_t i1){
        for (std::size_t i = i0; i < i1; ++i) g_out[i] += (1.0f / static_cast<float>(xin[i])) * static_cast<float>(g_in[i]);
      });
    }
  };
  return make_from_node(out);
}

Variable clamp(const Variable& X, float lo, float hi) {
  auto out = std::make_shared<Node>();
  out->shape = X.n->shape;
  out->value.resize(numel(out->shape));
  out->grad.resize(numel(out->shape), 0.0f);
  out->requires_grad = X.n->requires_grad;
  out->parents = {X.n};

  const std::size_t N = out->value.size();
  if (N < EW_SERIAL_CUTOFF) {
    for (std::size_t i = 0; i < N; ++i) {
      float v = static_cast<float>(X.n->value[i]);
      if (v < lo) v = lo;
      if (v > hi) v = hi;
      out->value[i] = v;
    }
  } else {
    const auto xin = X.n->value.data();
    auto outp = out->value.data();
    ag::parallel::parallel_for(N, EW_GRAIN, [&](std::size_t i0, std::size_t i1){
      for (std::size_t i = i0; i < i1; ++i) {
        float v = static_cast<float>(xin[i]);
        if (v < lo) v = lo;
        if (v > hi) v = hi;
        outp[i] = v;
      }
    });
  }

  out->backward = [Xn = X.n, lo, hi, oweak = std::weak_ptr<Node>(out)]() {
    auto op = oweak.lock(); if (!op) return;
    Node* o = op.get();
    if (!Xn || !Xn->requires_grad) return;
    if (Xn->grad.size() != Xn->value.size()) Xn->grad.assign(Xn->value.size(), 0.0f);

    const std::size_t N = o->value.size();
    ag::parallel::ScopedDeterministicParallel scope_guard;
    if (N < EW_SERIAL_CUTOFF) {
      for (std::size_t i = 0; i < N; ++i) {
        float v = static_cast<float>(Xn->value[i]);
        float g = (v <= lo || v >= hi) ? 0.0f : 1.0f;   // subgradient
        Xn->grad[i] += g * static_cast<float>(o->grad[i]);
      }
    } else {
      const auto xin = Xn->value.data();
      const auto g_in = o->grad.data();
      auto g_out = Xn->grad.data();
      ag::parallel::parallel_for(N, EW_GRAIN, [&](std::size_t i0, std::size_t i1){
        for (std::size_t i = i0; i < i1; ++i) {
          float v = static_cast<float>(xin[i]);
          float g = (v <= lo || v >= hi) ? 0.0f : 1.0f;
          g_out[i] += g * static_cast<float>(g_in[i]);
        }
      });
    }
  };
  return make_from_node(out);
}

} // namespace ag
