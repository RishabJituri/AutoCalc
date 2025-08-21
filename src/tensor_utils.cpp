#include "tensor_utils.hpp"
#include <algorithm>
#include <numeric>
#include <stdexcept>

namespace ag { 
namespace detail {

std::size_t numel(const std::vector<std::size_t>& shp) {
  return std::accumulate(shp.begin(), shp.end(), std::size_t{1}, std::multiplies<>());
}

std::vector<std::size_t> strides_for(const std::vector<std::size_t>& shp) {
  std::vector<std::size_t> st(shp.size(), 1);
  for (int i = int(shp.size()) - 2; i >= 0; --i)
    st[std::size_t(i)] = st[std::size_t(i + 1)] * shp[std::size_t(i + 1)];
  return st;
}

std::size_t ravel_index(const std::vector<std::size_t>& idx,
                        const std::vector<std::size_t>& strides) {
  std::size_t off = 0;
  for (std::size_t d = 0; d < idx.size(); ++d) off += idx[d] * strides[d];
  return off;
}

std::vector<std::size_t> unravel_index(std::size_t linear,
                                       const std::vector<std::size_t>& dims) {
  std::vector<std::size_t> idx(dims.size());
  for (int i = int(dims.size()) - 1; i >= 0; --i) {
    const auto ui = std::size_t(i);
    idx[ui] = dims[ui] ? (linear % dims[ui]) : 0;
    linear /= (dims[ui] ? dims[ui] : 1);
  }
  return idx;
}

std::vector<std::size_t> broadcast_two(const std::vector<std::size_t>& A,
                                       const std::vector<std::size_t>& B) {
  const std::size_t r = std::max(A.size(), B.size());
  std::vector<std::size_t> out(r, 1);
  for (std::size_t i = 0; i < r; ++i) {
    const std::size_t ad = (i < r - A.size()) ? 1 : A[i - (r - A.size())];
    const std::size_t bd = (i < r - B.size()) ? 1 : B[i - (r - B.size())];
    if (ad != bd && ad != 1 && bd != 1)
      throw std::invalid_argument("Incompatible shapes for broadcast");
    out[i] = std::max(ad, bd);
  }
  return out;
}

std::vector<std::size_t> broadcast_batch(const std::vector<std::size_t>& a_batch,
                                         const std::vector<std::size_t>& b_batch) {
  const std::size_t ra = a_batch.size();
  const std::size_t rb = b_batch.size();
  const std::size_t r  = std::max(ra, rb);
  std::vector<std::size_t> out(r, 1);
  for (std::size_t i = 0; i < r; ++i) {
    const std::size_t ad = (i < r - ra) ? 1 : a_batch[i - (r - ra)];
    const std::size_t bd = (i < r - rb) ? 1 : b_batch[i - (r - rb)];
    if (ad != bd && ad != 1 && bd != 1)
      throw std::invalid_argument("Incompatible batch dims for matmul");
    out[i] = std::max(ad, bd);
  }
  return out;
}

}
} // namespace ag::detail
