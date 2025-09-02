#pragma once
#include "ag/core/variables.hpp"
#include <vector>
#include <cstddef>

namespace ag {

// Sum over 'axes'. If axes is empty, reduce all dims. If keepdims=true, reduced
// axes are kept with size 1.
Variable reduce_sum(const Variable& x,
                    const std::vector<int>& axes = {},
                    bool keepdims = false);

// Mean over 'axes'. If axes is empty, reduce all dims. If keepdims=true, reduced
// axes are kept with size 1.
Variable reduce_mean(const Variable& x,
                     const std::vector<int>& axes = {},
                     bool keepdims = false);

// Broadcast x to target 'shape' (NumPy rules). The backward accumulates (sums)
// over broadcasted axes.
Variable broadcast_to(const Variable& x, const std::vector<std::size_t>& shape);

} // namespace ag
