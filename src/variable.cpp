// ============================================================================
// autograd.cpp  —  Implementation
// ============================================================================
#include "autograd.hpp"

#include <algorithm>
#include <numeric>
#include <stdexcept>
#include <unordered_set>

namespace ag {
namespace detail {

inline size_t s(const std::vector<size_t>& shp) {
  return std::accumulate(shp.begin(), shp.end(), static_cast<size_t>(1), std::multiplies<size_t>());
}

inline std::vector<size_t> strides_for(const std::vector<size_t>& shp) {
  std::vector<size_t> st(shp.size(), 1);
  for (int i = static_cast<int>(shp.size()) - 2; i >= 0; --i) {
    st[static_cast<size_t>(i)] = st[static_cast<size_t>(i + 1)] * shp[static_cast<size_t>(i + 1)];
  }
  return st;
}

inline size_t ravel_index(const std::vector<size_t>& idx, const std::vector<size_t>& strides) {
  size_t off = 0;
  for (size_t d = 0; d < idx.size(); ++d) off += idx[d] * strides[d];
  return off;
}

inline std::vector<size_t> unravel_index(size_t linear, const std::vector<size_t>& dims) {
  std::vector<size_t> idx(dims.size());
  for (int i = static_cast<int>(dims.size()) - 1; i >= 0; --i) {
    const size_t ui = static_cast<size_t>(i);
    idx[ui] = linear % dims[ui];
    linear /= dims[ui];
  }
  return idx;
}

inline std::vector<size_t> broadcast_batch(const std::vector<size_t>& a_batch,
                                           const std::vector<size_t>& b_batch) {
  const size_t ra = a_batch.size();
  const size_t rb = b_batch.size();
  const size_t r  = std::max(ra, rb);
  std::vector<size_t> out(r, 1);
  for (size_t i = 0; i < r; ++i) {
    const size_t ad = (i < r - ra) ? 1 : a_batch[i - (r - ra)];
    const size_t bd = (i < r - rb) ? 1 : b_batch[i - (r - rb)];
    if (ad != bd && ad != 1 && bd != 1) throw std::invalid_argument("Incompatible batch dims for matmul");
    out[i] = std::max(ad, bd);
  }
  return out;
}

// DFS topological order (parents first)
inline void topo_collect(const std::shared_ptr<Node>& node,
                         std::vector<std::shared_ptr<Node>>& order,
                         std::unordered_set<Node*>& seen) {
  if (!node || seen.count(node.get())) return;
  seen.insert(node.get());
  for (auto& p : node->parents) topo_collect(p, order, seen);
  order.push_back(node);
}

} // namespace detail

// -----------------------------
// Variable
// -----------------------------
Variable::Variable() : n(std::make_shared<Node>()) {}

Variable::Variable(std::shared_ptr<Node> node) : n(std::move(node)) {}

Variable::Variable(const std::vector<double>& value,
                   const std::vector<size_t>&  shape,
                   bool                        req) {
  if (detail::numel(shape) != value.size()) throw std::invalid_argument("value size != numel(shape)");
  n                 = std::make_shared<Node>();
  n->value          = value;
  n->shape          = shape;
  n->requires_grad  = req;
  n->grad.assign(value.size(), 0.0);
}

void Variable::zero_grad() {
  std::vector<std::shared_ptr<Node>> order;
  std::unordered_set<Node*>          seen;
  detail::topo_collect(n, order, seen);
  for (auto& x : order) x->grad.assign(detail::numel(x->shape), 0.0);
}

void Variable::backward() {
  using detail::numel;
  if (numel(n->shape) != 1) throw std::invalid_argument("Non-scalar output: pass seed gradient to backward(seed)");
  backward(std::vector<double>{1.0});
}

void Variable::backward(const std::vector<double>& seed) {
  // Collect subgraph
  std::vector<std::shared_ptr<Node>> order;
  std::unordered_set<Node*>          seen;
  detail::topo_collect(n, order, seen);

  // Seed gradient at output
  if (seed.size() != n->grad.size()) throw std::invalid_argument("Seed grad size mismatch");
  for (size_t i = 0; i < seed.size(); ++i) n->grad[i] += seed[i];

  // Reverse traversal
  for (auto it = order.rbegin(); it != order.rend(); ++it) {
    auto& node = *it;
    if (node->backward) node->backward();
  }
}

Variable scalar(double v, bool requires_grad) {
  return Variable(std::vector<double>{v}, std::vector<size_t>{1}, requires_grad);
}

// -----------------------------
// matmul (batched)
// -----------------------------
Variable matmul(const Variable& A, const Variable& B) {
  const auto& a_shape = A.n->shape;
  const auto& b_shape = B.n->shape;
  if (a_shape.size() < 2 || b_shape.size() < 2) throw std::invalid_argument("matmul requires rank >= 2 tensors");

  const size_t M  = a_shape[a_shape.size() - 2];
  const size_t K1 = a_shape[a_shape.size() - 1];
  const size_t K2 = b_shape[b_shape.size() - 2];
  const size_t N  = b_shape[b_shape.size() - 1];
  if (K1 != K2) throw std::invalid_argument("Inner dimensions must match for matmul");

  const std::vector<size_t> a_batch(a_shape.begin(), a_shape.end() - 2);
  const std::vector<size_t> b_batch(b_shape.begin(), b_shape.end() - 2);
  const std::vector<size_t> batch = detail::broadcast_batch(a_batch, b_batch);

  // Output shape
  std::vector<size_t> out_shape = batch;
  out_shape.push_back(M);
  out_shape.push_back(N);

  const auto  a_strides = detail::strides_for(a_shape);
  const auto  b_strides = detail::strides_for(b_shape);
  const auto  o_strides = detail::strides_for(out_shape);
  const size_t out_elems = detail::numel(out_shape);

  // Create output node
  auto out          = std::make_shared<Node>();
  out->value.assign(out_elems, 0.0);
  out->shape         = out_shape;
  out->grad.assign(out_elems, 0.0);
  out->requires_grad = (A.n->requires_grad || B.n->requires_grad);
  out->parents       = {A.n, B.n};

  // Forward pass
  const size_t batch_elems = batch.empty() ? 1 : detail::numel(batch);
  for (size_t b_lin = 0; b_lin < batch_elems; ++b_lin) {
    const auto b_idx = detail::unravel_index(b_lin, batch);

    // Map broadcasted batch index to A and B batch indices (right-aligned)
    std::vector<size_t> a_bidx(a_batch.size(), 0), b_bidx(b_batch.size(), 0);
    for (size_t i = 0; i < batch.size(); ++i) {
      const size_t bi = b_idx[i];
      if (i >= batch.size() - a_batch.size()) {
        const size_t ai = i - (batch.size() - a_batch.size());
        a_bidx[ai]      = (a_batch[ai] == 1) ? 0 : bi;
      }
      if (i >= batch.size() - b_batch.size()) {
        const size_t bj = i - (batch.size() - b_batch.size());
        b_bidx[bj]      = (b_batch[bj] == 1) ? 0 : bi;
      }
    }

    const size_t a_base = a_bidx.empty() ? 0 : detail::ravel_index(a_bidx, a_strides);
    const size_t b_base = b_bidx.empty() ? 0 : detail::ravel_index(b_bidx, b_strides);
    const size_t o_base = b_idx.empty() ? 0 : detail::ravel_index(b_idx, o_strides);

    for (size_t i = 0; i < M; ++i) {
      for (size_t j = 0; j < N; ++j) {
        double s = 0.0;
        for (size_t k = 0; k < K1; ++k) {
          const size_t a_off = a_base + i * a_strides[a_shape.size() - 2] + k * a_strides[a_shape.size() - 1];
          const size_t b_off = b_base + k * b_strides[b_shape.size() - 2] + j * b_strides[b_shape.size() - 1];
          s += A.n->value[a_off] * B.n->value[b_off];
        }
        const size_t o_off = o_base + i * o_strides[out_shape.size() - 2] + j * o_strides[out_shape.size() - 1];
        out->value[o_off]  = s;
      }
    }
  }

  // Backward (dA = dOut @ B^T, dB = A^T @ dOut), with batch broadcasting
  std::weak_ptr<Node> out_w = out, a_w = A.n, b_w = B.n;
  out->backward = [out_w, a_w, b_w, a_shape, b_shape, out_shape, a_strides, b_strides, o_strides]() {
    auto out_n = out_w.lock();
    if (!out_n) return;
    auto an = a_w.lock();
    auto bn = b_w.lock();
    if ((!an || !an->requires_grad) && (!bn || !bn->requires_grad)) return;

    const std::vector<size_t> a_batch(a_shape.begin(), a_shape.end() - 2);
    const std::vector<size_t> b_batch(b_shape.begin(), b_shape.end() - 2);
    const std::vector<size_t> batch = detail::broadcast_batch(a_batch, b_batch);

    const size_t M = a_shape[a_shape.size() - 2];
    const size_t K = a_shape[a_shape.size() - 1];
    const size_t N = b_shape[b_shape.size() - 1];

    const size_t batch_elems = batch.empty() ? 1 : detail::numel(batch);

    for (size_t b_lin = 0; b_lin < batch_elems; ++b_lin) {
      const auto b_idx = detail::unravel_index(b_lin, batch);

      std::vector<size_t> a_bidx(a_batch.size(), 0), b_bidx(b_batch.size(), 0);
      for (size_t i = 0; i < batch.size(); ++i) {
        const size_t bi = b_idx[i];
        if (i >= batch.size() - a_batch.size()) {
          const size_t ai = i - (batch.size() - a_batch.size());
          a_bidx[ai]      = (a_batch[ai] == 1) ? 0 : bi; // broadcast reduction
        }
        if (i >= batch.size() - b_batch.size()) {
          const size_t bj = i - (batch.size() - b_batch.size());
          b_bidx[bj]      = (b_batch[bj] == 1) ? 0 : bi;
        }
      }

      const size_t a_base = a_bidx.empty() ? 0 : detail::ravel_index(a_bidx, a_strides);
      const size_t b_base = b_bidx.empty() ? 0 : detail::ravel_index(b_bidx, b_strides);
      const size_t o_base = b_idx.empty() ? 0 : detail::ravel_index(b_idx, o_strides);

      for (size_t i = 0; i < M; ++i) {
        for (size_t j = 0; j < N; ++j) {
          const size_t o_off = o_base + i * o_strides[out_shape.size() - 2] + j * o_strides[out_shape.size() - 1];
          const double go    = out_n->grad[o_off]; // dL/dOut[i,j]

          if (an && an->requires_grad) {
            for (size_t k = 0; k < K; ++k) {
              const size_t a_off = a_base + i * a_strides[a_shape.size() - 2] + k * a_strides[a_shape.size() - 1];
              const size_t b_off = b_base + k * b_strides[b_shape.size() - 2] + j * b_strides[b_shape.size() - 1];
              an->grad[a_off] += go * (bn ? bn->value[b_off] : 0.0); // dA += dOut * B^T
            }
          }

          if (bn && bn->requires_grad) {
            for (size_t k = 0; k < K; ++k) {
              const size_t a_off = a_base + i * a_strides[a_shape.size() - 2] + k * a_strides[a_shape.size() - 1];
              const size_t b_off = b_base + k * b_strides[b_shape.size() - 2] + j * b_strides[b_shape.size() - 1];
              bn->grad[b_off] += go * (an ? an->value[a_off] : 0.0); // dB += A^T * dOut
            }
          }
        }
      }
    }
  };

  // Wrap the node to preserve graph/backward.
  return Variable(out);
}

} // namespace ag
