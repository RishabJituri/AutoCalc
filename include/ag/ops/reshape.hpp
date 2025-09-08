#pragma once
#include "ag/core/variables.hpp"
#include <cstddef>


namespace ag {

// Flatten starting at start_dim into a 2D [prefix, suffix] then drop the extra axis.
// For MNIST, commonly flatten(x, 1) to go from [B,C,H,W] -> [B, C*H*W]
Variable flatten(const Variable& x, std::size_t start_dim = 1);

Variable reshape(const Variable& X, const std::vector<std::size_t>& new_shape);

} // namespace ag
