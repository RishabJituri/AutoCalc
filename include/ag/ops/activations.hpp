#pragma once
#include "ag/core/variables.hpp"

namespace ag {

// Elementwise activations (same shape as input)
Variable relu(const Variable& x);
Variable sigmoid(const Variable& x);
Variable tanhv(const Variable& x);   // named tanhv to avoid clash with std::tanh
Variable logv(const Variable& x);
Variable clamp(const Variable& x, float lo, float hi);

} // namespace ag
