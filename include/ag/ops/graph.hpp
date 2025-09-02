#pragma once
#include "ag/core/variables.hpp"

namespace ag {
// Returns a detached copy: same values/shape, no parents, requires_grad=false.
Variable stop_gradient(const Variable& x);

Variable detach(const Variable& x);
}

