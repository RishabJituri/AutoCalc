#pragma once
#include "ag/core/variables.hpp"
#include <vector>

namespace ag {

// General N-D transpose (permute axes). axes is a permutation of [0..rank-1].
Variable transpose(const Variable& X, const std::vector<int>& axes);

// 2-D convenience: swap last two dimensions (or for rank==2, swap 0 and 1).
Variable t(const Variable& X);

} // namespace ag
