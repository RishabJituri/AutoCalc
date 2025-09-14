#pragma once
#include <cstddef>
#include <cstdint>
#include "ag/sys/hw.hpp"

namespace ag::parallel {

// False-sharing-proof slot for per-thread scratch (one whole cache line).
template <class T, std::size_t Line = ag::sys::kCacheLine>
struct alignas(Line) PerThreadSlot {
  T v;
  unsigned char _pad[(Line - (sizeof(T) % Line)) % Line];
};

// Small helper to compute array size from thread count.
inline std::size_t slots_for_threads(std::size_t threads) { return threads ? threads : 1; }

} // namespace ag::parallel
