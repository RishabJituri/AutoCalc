#pragma once
#include "ag/core/variables.hpp"

namespace ag {

// Elementwise (broadcasting)
Variable add(const Variable& a, const Variable& b);
Variable sub(const Variable& a, const Variable& b);
Variable mul(const Variable& a, const Variable& b);
Variable div(const Variable& a, const Variable& b);
Variable neg(const Variable& x);
// Trig/exp variants in this codebase are named with 'v' suffix
Variable sinv(const Variable& x);
Variable cosv(const Variable& x);
Variable expv(const Variable& x);
Variable pow(const Variable& base, const Variable& exponent);

} // namespace ag
