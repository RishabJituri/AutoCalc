#include "ag/ops/reshape.hpp"
#include "ag/ops/tensor_utils.hpp"
#include "ag/parallel/parallel_for.hpp"
#include <numeric>

namespace ag {
using detail::numel;

Variable flatten(const Variable& X, std::size_t start_dim) {
  const auto& shp = X.n->shape;
  const std::size_t rank = shp.size();
  if (start_dim >= rank) return X; // no-op

  std::size_t prefix = 1;
  for (std::size_t i = 0; i < start_dim; ++i) prefix *= shp[i];
  std::size_t suffix = 1;
  for (std::size_t i = start_dim; i < rank; ++i) suffix *= shp[i];

  auto out = std::make_shared<Node>();
  out->shape = {prefix, suffix};

  const std::size_t N = prefix * suffix;
  out->value.resize(N);

  // Forward copy: parallel for large tensors
  const std::size_t COPY_SERIAL_CUTOFF = 4096;
  const std::size_t COPY_GRAIN = 1024;
  if (N < COPY_SERIAL_CUTOFF) {
    std::copy(X.n->value.begin(), X.n->value.end(), out->value.begin());
  } else {
    const auto xin = X.n->value.data();
    auto outp = out->value.data();
    ag::parallel::parallel_for(N, COPY_GRAIN, [&](std::size_t i0, std::size_t i1){
      for (std::size_t i = i0; i < i1; ++i) outp[i] = xin[i];
    });
  }

  out->grad.assign(N, 0.0);
  out->requires_grad = X.n->requires_grad;
  out->parents = {X.n};

  out->backward = [Xn = X.n, oweak = std::weak_ptr<ag::Node>(out)]() {
    auto op = oweak.lock(); if (!op) return;
    ag::Node* o = op.get();

    if (!Xn || !Xn->requires_grad) return;
    if (Xn->grad.size() != Xn->value.size()) Xn->grad.assign(Xn->value.size(), 0.0f);

    const std::size_t N = o->grad.size();
    const std::size_t COPY_SERIAL_CUTOFF = 4096;
    const std::size_t COPY_GRAIN = 1024;
    if (N < COPY_SERIAL_CUTOFF) {
      for (std::size_t i = 0; i < N; ++i) Xn->grad[i] += o->grad[i];
    } else {
      const auto gin = o->grad.data();
      auto gout = Xn->grad.data();
      ag::parallel::parallel_for(N, COPY_GRAIN, [&](std::size_t i0, std::size_t i1){
        for (std::size_t i = i0; i < i1; ++i) gout[i] += static_cast<float>(gin[i]);
      });
    }
  };
  return make_from_node(out);
}

} // namespace ag


// --- appended: general reshape ---
#include <stdexcept>

namespace ag {

Variable reshape(const Variable& X, const std::vector<std::size_t>& new_shape) {
  const auto old_elems = detail::numel(X.n->shape);
  const auto new_elems = detail::numel(new_shape);
  if (old_elems != new_elems) {
    throw std::runtime_error("reshape: number of elements must remain constant");
  }
  if (new_shape == X.n->shape) {  // no-op view
    return X;
  }

  auto out = std::make_shared<Node>();
  out->shape = new_shape;

  const std::size_t N = new_elems;
  out->value.resize(N);

  // Forward copy (parallel when large)
  const std::size_t COPY_SERIAL_CUTOFF = 4096;
  const std::size_t COPY_GRAIN = 1024;
  if (N < COPY_SERIAL_CUTOFF) {
    std::copy(X.n->value.begin(), X.n->value.end(), out->value.begin());
  } else {
    const auto xin = X.n->value.data();
    auto outp = out->value.data();
    ag::parallel::parallel_for(N, COPY_GRAIN, [&](std::size_t i0, std::size_t i1){
      for (std::size_t i = i0; i < i1; ++i) outp[i] = xin[i];
    });
  }

  out->grad.assign(N, 0.0);
  out->requires_grad = X.n->requires_grad;
  out->parents = {X.n};

  out->backward = [Xn = X.n, oweak = std::weak_ptr<ag::Node>(out)]() {
    auto op = oweak.lock(); if (!op) return;
    ag::Node* o = op.get();
    if (!Xn || !Xn->requires_grad) return;
    if (Xn->grad.size() != Xn->value.size()) Xn->grad.assign(Xn->value.size(), 0.0f);

    const std::size_t N = o->grad.size();
    const std::size_t COPY_SERIAL_CUTOFF = 4096;
    const std::size_t COPY_GRAIN = 1024;
    if (N < COPY_SERIAL_CUTOFF) {
      for (std::size_t i = 0; i < N; ++i) Xn->grad[i] += o->grad[i];
    } else {
      const auto gin = o->grad.data();
      auto gout = Xn->grad.data();
      ag::parallel::parallel_for(N, COPY_GRAIN, [&](std::size_t i0, std::size_t i1){
        for (std::size_t i = i0; i < i1; ++i) gout[i] += static_cast<float>(gin[i]);
      });
    }
  };
  return make_from_node(out);
}

// Convenience overload for initializer_list
Variable reshape(const Variable& X, std::initializer_list<std::size_t> new_shape) {
  return reshape(X, std::vector<std::size_t>(new_shape));
}

} // namespace ag
