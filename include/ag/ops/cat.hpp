// filepath: include/ag/ops/cat.hpp
#pragma once
#include "ag/core/variables.hpp"
#include <vector>
#include <cstddef>

namespace ag {

// Concatenate Variables along the given axis.
// All inputs must have the same rank and matching dimensions on all axes except `axis`.
// Returns a new Variable whose `axis` dimension equals the sum of the inputs' axis dimensions.
// Supports autograd: backward splits the upstream gradient and routes slices to each input.
Variable cat(const std::vector<Variable>& inputs, int axis = 0);

} // namespace ag
