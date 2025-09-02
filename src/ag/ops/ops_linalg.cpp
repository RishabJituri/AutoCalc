#include "ag/core/variables.hpp"
#include "ag/ops/tensor_utils.hpp"
#include <stdexcept>

namespace ag {
using detail::numel;
using detail::strides_for;
using detail::unravel_index;
using detail::ravel_index;
using detail::broadcast_batch;

Variable matmul(const Variable& A, const Variable& B) {
  const auto& a_shape = A.n->shape;
  const auto& b_shape = B.n->shape;
  if (a_shape.size() < 2 || b_shape.size() < 2)
    throw std::invalid_argument("matmul requires rank >= 2 tensors");

  const std::size_t M  = a_shape[a_shape.size()-2];
  const std::size_t K1 = a_shape[a_shape.size()-1];
  const std::size_t K2 = b_shape[b_shape.size()-2];
  const std::size_t N  = b_shape[b_shape.size()-1];
  if (K1 != K2) throw std::invalid_argument("Inner dimensions must match for matmul");

  const std::vector<std::size_t> a_batch(a_shape.begin(), a_shape.end()-2);
  const std::vector<std::size_t> b_batch(b_shape.begin(), b_shape.end()-2);
  const std::vector<std::size_t> batch = broadcast_batch(a_batch, b_batch);

  std::vector<std::size_t> out_shape = batch; out_shape.push_back(M); out_shape.push_back(N);
  const auto outN = numel(out_shape);

  auto out = std::make_shared<Node>();
  out->value.assign(outN, 0.0); out->grad.assign(outN, 0.0);
  out->shape = out_shape; out->parents = {A.n, B.n};
  out->requires_grad = (A.n->requires_grad || B.n->requires_grad);

  const auto a_str = strides_for(a_shape);
  const auto b_str = strides_for(b_shape);
  const auto o_str = strides_for(out_shape);

  const std::size_t batch_elems = batch.empty()? 1 : numel(batch);
  for (std::size_t blin = 0; blin < batch_elems; ++blin) {
    const auto bidx = unravel_index(blin, batch);
    std::vector<std::size_t> a_bidx(a_batch.size(),0), b_bidx(b_batch.size(),0);
    for (std::size_t i=0;i<batch.size();++i) {
      const std::size_t bi = bidx[i];
      if (i >= batch.size()-a_batch.size()) {
        const std::size_t ai = i-(batch.size()-a_batch.size());
        a_bidx[ai] = (a_batch[ai]==1)? 0 : bi;
      }
      if (i >= batch.size()-b_batch.size()) {
        const std::size_t bj = i-(batch.size()-b_batch.size());
        b_bidx[bj] = (b_batch[bj]==1)? 0 : bi;
      }
    }
    const std::size_t a_base = a_bidx.empty()? 0 : ravel_index(a_bidx, a_str);
    const std::size_t b_base = b_bidx.empty()? 0 : ravel_index(b_bidx, b_str);
    const std::size_t o_base = bidx.empty()? 0 : ravel_index(bidx, o_str);

    for (std::size_t i=0;i<M;++i) {
      for (std::size_t j=0;j<N;++j) {
        double s = 0.0;
        for (std::size_t k=0;k<K1;++k) {
          const auto a_off = a_base + i*a_str[a_shape.size()-2] + k*a_str[a_shape.size()-1];
          const auto b_off = b_base + k*b_str[b_shape.size()-2] + j*b_str[b_shape.size()-1];
          s += A.n->value[a_off] * B.n->value[b_off];
        }
        const auto o_off = o_base + i*o_str[out_shape.size()-2] + j*o_str[out_shape.size()-1];
        out->value[o_off] = s;
      }
    }
  }

  std::weak_ptr<Node> ow = out, aw = A.n, bw = B.n;
  out->backward = [ow, aw, bw, a_shape, b_shape, out_shape, a_str, b_str, o_str]() {
    auto o = ow.lock(); if (!o) return; auto a = aw.lock(); auto b = bw.lock();
    if ((!a || !a->requires_grad) && (!b || !b->requires_grad)) return;

    const std::vector<std::size_t> a_batch(a_shape.begin(), a_shape.end()-2);
    const std::vector<std::size_t> b_batch(b_shape.begin(), b_shape.end()-2);
    const std::vector<std::size_t> batch = broadcast_batch(a_batch, b_batch);

    const std::size_t M=a_shape[a_shape.size()-2], K=a_shape[a_shape.size()-1], N=b_shape[b_shape.size()-1];
    const std::size_t batch_elems = batch.empty()? 1 : numel(batch);

    for (std::size_t blin = 0; blin < batch_elems; ++blin) {
      const auto bidx = detail::unravel_index(blin, batch);
      std::vector<std::size_t> a_bidx(a_batch.size(),0), b_bidx(b_batch.size(),0);
      for (std::size_t i=0;i<batch.size();++i) {
        const std::size_t bi = bidx[i];
        if (i >= batch.size()-a_batch.size()) {
          const std::size_t ai = i-(batch.size()-a_batch.size());
          a_bidx[ai] = (a_batch[ai]==1)? 0 : bi;
        }
        if (i >= batch.size()-b_batch.size()) {
          const std::size_t bj = i-(batch.size()-b_batch.size());
          b_bidx[bj] = (b_batch[bj]==1)? 0 : bi;
        }
      }
      const std::size_t a_base = a_bidx.empty()? 0 : ravel_index(a_bidx, a_str);
      const std::size_t b_base = b_bidx.empty()? 0 : ravel_index(b_bidx, b_str);
      const std::size_t o_base = bidx.empty()? 0 : ravel_index(bidx, o_str);

      for (std::size_t i=0;i<M;++i) {
        for (std::size_t j=0;j<N;++j) {
          const std::size_t o_off = o_base + i*o_str[out_shape.size()-2] + j*o_str[out_shape.size()-1];
          const double go = o->grad[o_off];
          if (a && a->requires_grad) {
            for (std::size_t k=0;k<K;++k) {
              const auto a_off = a_base + i*a_str[a_shape.size()-2] + k*a_str[a_shape.size()-1];
              const auto b_off = b_base + k*b_str[b_shape.size()-2] + j*b_str[b_shape.size()-1];
              a->grad[a_off] += go * (b ? b->value[b_off] : 0.0); // dA += dC * B^T
            }
          }
          if (b && b->requires_grad) {
            for (std::size_t k=0;k<K;++k) {
              const auto a_off = a_base + i*a_str[a_shape.size()-2] + k*a_str[a_shape.size()-1];
              const auto b_off = b_base + k*b_str[b_shape.size()-2] + j*b_str[b_shape.size()-1];
              b->grad[b_off] += go * (a ? a->value[a_off] : 0.0); // dB += A^T * dC
            }
          }
        }
      }
    }
  };

  return make_from_node(out);
}

} // namespace ag
