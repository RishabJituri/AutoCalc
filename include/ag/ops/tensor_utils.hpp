#pragma once
#include <vector>
#include <cstddef>

namespace ag { 
namespace detail {

// shape helpers
std::size_t numel(const std::vector<std::size_t>& shp);
std::vector<std::size_t> strides_for(const std::vector<std::size_t>& shp);
std::size_t ravel_index(const std::vector<std::size_t>& idx,
                        const std::vector<std::size_t>& strides);
std::vector<std::size_t> unravel_index(std::size_t linear,
                                       const std::vector<std::size_t>& dims);

// broadcasting helpers (NumPy-style, right-aligned; 1 is wildcard)
std::vector<std::size_t> broadcast_two(const std::vector<std::size_t>& A,
                                       const std::vector<std::size_t>& B);
std::vector<std::size_t> broadcast_batch(const std::vector<std::size_t>& a_batch,
                                         const std::vector<std::size_t>& b_batch);

}
} // namespace ag::detail
