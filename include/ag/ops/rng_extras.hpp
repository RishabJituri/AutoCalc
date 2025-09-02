#pragma once
#include "ag/core/rng.hpp"
#include <cstdint>

namespace ag {

// Thread-local global RNG accessor and seed control
RNG& global_rng();
void set_global_seed(uint64_t seed);
uint64_t get_global_seed();

} // namespace ag
