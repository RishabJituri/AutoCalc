// filepath: include/ag/nn/layers/upsample.hpp
#pragma once
#include "ag/core/variables.hpp"
#include <cstddef>

namespace ag {

// Nearest-neighbor 2× upsampling for NCHW tensors.
// scale_h and scale_w are integer scale factors (default 2).
// Forward repeats each spatial element; backward accumulates into the source pixel.
Variable upsample2d(const Variable& x, std::size_t scale_h = 2, std::size_t scale_w = 2);

} // namespace ag
