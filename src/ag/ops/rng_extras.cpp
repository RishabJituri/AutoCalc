#include "ag/ops/rng_extras.hpp"

namespace ag {

static thread_local RNG tl_rng{/*default seed*/ 88172645463393265ull};
static uint64_t last_seed = 88172645463393265ull;

RNG& global_rng() { return tl_rng; }

void set_global_seed(uint64_t seed) {
  if (seed == 0) seed = 88172645463393265ull;
  tl_rng = RNG(seed);
  last_seed = seed;
}

uint64_t get_global_seed() { return last_seed; }

} // namespace ag
