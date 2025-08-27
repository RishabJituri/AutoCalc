
#ifndef AG_RNG_HPP
#define AG_RNG_HPP

#include <cstdint>

namespace ag {

// Tiny xorshift64* RNG for portability (not cryptographic)
struct RNG {
  uint64_t state;
  explicit RNG(uint64_t seed = 88172645463393265ull) : state(seed?seed:88172645463393265ull) {}
  inline uint64_t next_u64() {
    uint64_t x = state;
    x ^= x >> 12;
    x ^= x << 25;
    x ^= x >> 27;
    state = x;
    return x * 2685821657736338717ull;
  }
  inline double next_uniform01() {
    // 53-bit mantissa -> [0,1)
    return (next_u64() >> 11) * (1.0/9007199254740992.0);
  }
};

} // namespace ag

#endif // AG_RNG_HPP
