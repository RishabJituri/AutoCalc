#pragma once
#include "ag/core/variables.hpp"
#include <vector>
#include <cstddef>

namespace ag::nn {

// Cross-entropy loss on logits: logits[B,C], targets[B] in [0..C-1]. Returns scalar mean loss.
Variable cross_entropy(const Variable& logits, const std::vector<std::size_t>& targets);

} // namespace ag::nn
