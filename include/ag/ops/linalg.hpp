#pragma once
#include "ag/core/variables.hpp"
#include <vector>
#include <cstdint>

namespace ag {

// Linear algebra
// Batched matmul: A:[B..., M, K] @ B:[C..., K, N] -> [broadcast(B...,C...), M, N]
Variable matmul(const Variable& A, const Variable& B);

} // namespace ag
