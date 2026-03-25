// filepath: src/ag/ops/cat.cpp
#include "ag/ops/cat.hpp"
#include "ag/ops/tensor_utils.hpp"
#include "ag/parallel/parallel_for.hpp"
#include <stdexcept>
#include <numeric>

namespace ag {
using detail::numel;
using detail::strides_for;

Variable cat(const std::vector<Variable>& inputs, int axis) {
  if (inputs.empty())
    throw std::invalid_argument("cat: inputs must be non-empty");

  const auto& ref_shape = inputs[0].n->shape;
  const std::size_t rank = ref_shape.size();
  if (rank == 0)
    throw std::invalid_argument("cat: cannot concatenate scalars");

  // Normalize negative axis
  int ax = axis;
  if (ax < 0) ax += static_cast<int>(rank);
  if (ax < 0 || ax >= static_cast<int>(rank))
    throw std::invalid_argument("cat: axis out of range");
  const std::size_t dim = static_cast<std::size_t>(ax);

  // Validate shapes: all ranks must match, all dims except `dim` must match
  std::size_t total_along_dim = 0;
  for (std::size_t t = 0; t < inputs.size(); ++t) {
    const auto& sh = inputs[t].n->shape;
    if (sh.size() != rank)
      throw std::invalid_argument("cat: all inputs must have the same rank");
    for (std::size_t d = 0; d < rank; ++d) {
      if (d != dim && sh[d] != ref_shape[d])
        throw std::invalid_argument("cat: shape mismatch on non-cat dimension");
    }
    total_along_dim += sh[dim];
  }

  // Build output shape
  std::vector<std::size_t> out_shape = ref_shape;
  out_shape[dim] = total_along_dim;
  const std::size_t out_numel = numel(out_shape);

  auto out = std::make_shared<Node>();
  out->shape = out_shape;
  out->value.resize(out_numel);
  out->grad.assign(out_numel, 0.0f);

  // Check requires_grad
  bool req = false;
  for (const auto& v : inputs) {
    out->parents.push_back(v.n);
    if (v.n->requires_grad) req = true;
  }
  out->requires_grad = req;

  // Forward: copy slices. Decompose indices into (outer, dim_idx, inner).
  //   outer  = product of dims before `dim`
  //   inner  = product of dims after  `dim`
  std::size_t outer = 1;
  for (std::size_t d = 0; d < dim; ++d) outer *= out_shape[d];
  std::size_t inner = 1;
  for (std::size_t d = dim + 1; d < rank; ++d) inner *= out_shape[d];

  // Precompute per-input dim sizes and cumulative offsets along the cat axis
  const std::size_t K = inputs.size();
  std::vector<std::size_t> inp_dim(K);    // size of dim `dim` for each input
  std::vector<std::size_t> offsets(K);    // cumulative offset along dim
  {
    std::size_t cum = 0;
    for (std::size_t t = 0; t < K; ++t) {
      offsets[t] = cum;
      inp_dim[t] = inputs[t].n->shape[dim];
      cum += inp_dim[t];
    }
  }

  // Copy data: for each input, copy its slice into the output
  float* out_ptr = out->value.data();
  for (std::size_t t = 0; t < K; ++t) {
    const float* in_ptr = inputs[t].n->value.data();
    const std::size_t d_t = inp_dim[t];
    const std::size_t off_t = offsets[t];
    const std::size_t out_dim_stride = total_along_dim * inner;
    const std::size_t in_dim_stride = d_t * inner;
    const std::size_t slice_bytes = d_t * inner;

    // Parallelize over outer dimension for large tensors
    if (outer * slice_bytes > 4096) {
      ag::parallel::parallel_for(outer, /*grain=*/1, [&](std::size_t o0, std::size_t o1) {
        for (std::size_t o = o0; o < o1; ++o) {
          const float* src = in_ptr + o * in_dim_stride;
          float* dst = out_ptr + o * out_dim_stride + off_t * inner;
          std::copy(src, src + d_t * inner, dst);
        }
      });
    } else {
      for (std::size_t o = 0; o < outer; ++o) {
        const float* src = in_ptr + o * in_dim_stride;
        float* dst = out_ptr + o * out_dim_stride + off_t * inner;
        std::copy(src, src + d_t * inner, dst);
      }
    }
  }

  // Backward: split the output gradient along `dim` and accumulate into each input's grad
  std::weak_ptr<Node> ow = out;
  // Capture input nodes and shape info
  std::vector<std::shared_ptr<Node>> inp_nodes;
  inp_nodes.reserve(K);
  for (const auto& v : inputs) inp_nodes.push_back(v.n);

  out->backward = [ow, inp_nodes, inp_dim, offsets, outer, inner, total_along_dim, K]() {
    auto o = ow.lock();
    if (!o) return;
    const float* grad_out = o->grad.data();

    for (std::size_t t = 0; t < K; ++t) {
      auto& inp = inp_nodes[t];
      if (!inp || !inp->requires_grad) continue;
      if (inp->grad.size() != inp->value.size())
        inp->grad.assign(inp->value.size(), 0.0f);

      float* grad_in = inp->grad.data();
      const std::size_t d_t = inp_dim[t];
      const std::size_t off_t = offsets[t];
      const std::size_t out_dim_stride = total_along_dim * inner;
      const std::size_t in_dim_stride = d_t * inner;
      const std::size_t slice_elems = d_t * inner;

      if (outer * slice_elems > 4096) {
        ag::parallel::parallel_for(outer, /*grain=*/1, [&](std::size_t o0, std::size_t o1) {
          for (std::size_t o = o0; o < o1; ++o) {
            const float* src = grad_out + o * out_dim_stride + off_t * inner;
            float* dst = grad_in + o * in_dim_stride;
            for (std::size_t i = 0; i < slice_elems; ++i)
              dst[i] += src[i];
          }
        });
      } else {
        for (std::size_t o = 0; o < outer; ++o) {
          const float* src = grad_out + o * out_dim_stride + off_t * inner;
          float* dst = grad_in + o * in_dim_stride;
          for (std::size_t i = 0; i < slice_elems; ++i)
            dst[i] += src[i];
        }
      }
    }
  };

  return make_from_node(out);
}

} // namespace ag
