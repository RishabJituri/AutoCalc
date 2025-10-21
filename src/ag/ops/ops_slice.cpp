#include <vector>
#include <stdexcept>
#include <numeric>
#include <functional>

#include "ag/core/variables.hpp"

namespace ag {

// Helper: compute strides for row-major layout
static std::vector<std::size_t> strides_for(const std::vector<std::size_t>& shape) {
  std::vector<std::size_t> st(shape.size(), 0);
  if (shape.empty()) return st;
  st.back() = 1;
  for (int i = int(shape.size()) - 2; i >= 0; --i) st[i] = st[i+1] * shape[i+1];
  return st;
}

Variable at(const Variable& A, const std::vector<std::size_t>& begin, const std::vector<std::size_t>& end) {
  const auto Ash = A.shape();
  const size_t r = Ash.size();
  if (begin.size() != r || end.size() != r) throw std::invalid_argument("at: begin/end must match rank");

  // Compute full output shape and validate ranges
  std::vector<std::size_t> OutSh_full(r);
  std::size_t out_elems = 1;
  for (size_t i = 0; i < r; ++i) {
    if (end[i] <= begin[i] || end[i] > Ash[i]) throw std::invalid_argument("at: invalid slice range");
    OutSh_full[i] = end[i] - begin[i];
    out_elems *= OutSh_full[i];
  }

  // Build reduced shape by removing unit-dimensions (integer-index semantics)
  std::vector<std::size_t> OutSh_reduced;
  for (size_t i = 0; i < r; ++i) {
    if (OutSh_full[i] != 1) OutSh_reduced.push_back(OutSh_full[i]);
  }

  // If all dims removed, reduced shape is empty (scalar)
  // Prepare output storage: number of elements equals product(OutSh_full) which is same as product(OutSh_reduced)
  std::vector<float> out(out_elems);

  // Compute strides and copy elements from A to out in row-major order (iterate over full-output layout)
  auto Astr = strides_for(Ash);
  auto Ostr_full = strides_for(OutSh_full);

  const float* Aptr = A.value().data();
  float* Outptr = out.data();

  // Iterate over output linear index and map to input
  for (std::size_t oi = 0; oi < out_elems; ++oi) {
    // unravel oi into multi-index (full layout)
    std::size_t rem = oi;
    std::size_t in_idx = 0;
    for (size_t d = 0; d < r; ++d) {
      const std::size_t od = Ostr_full[d] > 0 ? ( (rem / Ostr_full[d]) % OutSh_full[d] ) : 0;
      rem = rem % Ostr_full[d];
      const std::size_t id = begin[d] + od;
      in_idx += id * Astr[d];
    }
    Outptr[oi] = Aptr[in_idx];
  }

  const bool req = A.requires_grad() && ag::is_grad_enabled();
  // Use the reduced shape for the Variable's visible shape so integer-index dims are removed
  Variable C(out, OutSh_reduced, req);

  if (req) {
    C.n->parents = { A.n };
    // Capture full shapes and begin so we can map reduced-output grads back into the input
    C.n->backward = [An = A.n, Cn = C.n, Ash, OutSh_full, begin]() {
      if (An->grad.size() != An->value.size()) An->grad.assign(An->value.size(), 0.0f);
      const float* dC = Cn->grad.data();
      auto Astr = strides_for(Ash);
      auto Ostr = strides_for(OutSh_full);
      const size_t r = Ash.size();
      const size_t out_elems = dC ? std::accumulate(OutSh_full.begin(), OutSh_full.end(), size_t(1), std::multiplies<size_t>()) : 0;
      for (size_t oi = 0; oi < out_elems; ++oi) {
        size_t rem = oi;
        size_t in_idx = 0;
        for (size_t d = 0; d < r; ++d) {
          const std::size_t od = Ostr[d] > 0 ? ( (rem / Ostr[d]) % OutSh_full[d] ) : 0;
          rem = rem % Ostr[d];
          const std::size_t id = begin[d] + od;
          in_idx += id * Astr[d];
        }
        An->grad[in_idx] += dC[oi];
      }
    };
  }

  return C;
}

} // namespace ag
