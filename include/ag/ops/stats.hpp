#pragma once
#include "ag/core/variables.hpp"
#include <vector>

namespace ag {

// Reduce max over axes (NumPy-style). If axes empty -> reduce all.
// If keepdims true, keep reduced dims with size 1.
Variable reduce_max(const Variable& x,
                    const std::vector<int>& axes = {},
                    bool keepdims = false);

// LogSumExp over axes (numerically stable via max-shift).
Variable logsumexp(const Variable& x,
                   const std::vector<int>& axes = {},
                   bool keepdims = false);

// Softmax along a single axis (default last axis). Returns same shape as x.
Variable softmax(const Variable& x, int axis = -1);

// Argmax along an axis. No grad; returns plain indices vector for last axis per leading dims.
std::vector<std::size_t> argmax_lastdim(const Variable& x);

} // namespace ag
