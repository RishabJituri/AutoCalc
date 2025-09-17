
#ifndef AG_CORE_STOCHASTIC_HPP
#define AG_CORE_STOCHASTIC_HPP

#include <vector>
#include <cstddef>
#include <cstdint>
#include "ag/core/variables.hpp"

namespace ag {


struct SampleOut {
  Variable sample_onehot;  // shape [B,K] or [K], requires_grad=false
  Variable logprob;        // shape [B] or [], connected to logits for surrogate gradients
};

// Draw categorical samples (per row) from logits. Returns one-hot sample and log-prob Variable.
// Shapes supported: [K] or [B,K].
SampleOut CategoricalSample(const Variable& logits, uint64_t seed=123456789ull);

// Differentiable Gumbel-Softmax (Concrete) relaxation. Returns soft probs of same shape as logits.
// If 'hard' is true, uses straight-through: hard one-hot in forward, soft in backward.
Variable GumbelSoftmax(const Variable& logits, float tau=1.0, bool hard=false, uint64_t seed=987654321ull);
}
 // namespace ag

#endif // AG_STOCHASTIC_HPP
