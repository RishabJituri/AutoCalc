#pragma once
#include <vector>
#include <cstddef>

namespace ag {

// Lightweight slice/spec descriptor used by tensor utilities and bindings.
// Mirrors Python slice semantics: is_index==true means this spec is a single
// integer index stored in `start`. Otherwise start/stop/step describe the
// half-open range [start, stop) with step.
struct Slice {
  bool is_index = false;
  long long start = 0;
  long long stop = 0;
  long long step = 1;
};

class Variable; // forward

// High-level slicing API that accepts a per-dimension Slice spec.
Variable at(const Variable& X, const std::vector<Slice>& spec);

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
