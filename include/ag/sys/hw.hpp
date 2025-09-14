#pragma once
#include <cstddef>
#include <cstdint>

namespace ag::sys {

// Basic CPU cache facts (safe defaults if probing fails).
struct CacheInfo {
  std::size_t line = 64;       // cache line size in bytes
  std::size_t l1d  = 32u << 10; // L1 data cache bytes
  std::size_t l2   = 512u << 10; // L2 cache bytes (per-core or cluster)
};

// Detect once (thread-safe). Always returns non-zero fields.
const CacheInfo& cache_info() noexcept;

// Re-run detection and env parsing (mainly for tests/bench).
void reload() noexcept;

// Conservative cache-line constant (used for padding/alignment)
inline constexpr std::size_t kCacheLine = 64;

// Round 'bytes' up to the next cache-line multiple.
inline std::size_t pad_to_cacheline(std::size_t bytes) noexcept {
  return (bytes + (kCacheLine - 1)) & ~(kCacheLine - 1);
}

// Prefetch locality hint (0..3). Safe no-ops where unsupported.
enum class PrefetchLocality : int { None = 0, Low = 1, Med = 2, High = 3 };

// Replace prefetch helpers with this switch-based version:

template <class T>
inline void prefetch_read(const T* p,
                          PrefetchLocality l = PrefetchLocality::Med) noexcept {
#if defined(__GNUC__) || defined(__clang__)
  switch (l) {
    case PrefetchLocality::None: __builtin_prefetch((const void*)p, 0, 0); break;
    case PrefetchLocality::Low:  __builtin_prefetch((const void*)p, 0, 1); break;
    case PrefetchLocality::Med:  __builtin_prefetch((const void*)p, 0, 2); break;
    case PrefetchLocality::High: __builtin_prefetch((const void*)p, 0, 3); break;
  }
#else
  (void)p; (void)l;
#endif
}

template <class T>
inline void prefetch_write(const T* p,
                           PrefetchLocality l = PrefetchLocality::Med) noexcept {
#if defined(__GNUC__) || defined(__clang__)
  switch (l) {
    case PrefetchLocality::None: __builtin_prefetch((const void*)p, 1, 0); break;
    case PrefetchLocality::Low:  __builtin_prefetch((const void*)p, 1, 1); break;
    case PrefetchLocality::Med:  __builtin_prefetch((const void*)p, 1, 2); break;
    case PrefetchLocality::High: __builtin_prefetch((const void*)p, 1, 3); break;
  }
#else
  (void)p; (void)l;
#endif
}


// L1D-based default grain for 1-D loops (≈ L1D/4 elems). Clamp to >= 1.
std::size_t default_grain_for(std::size_t elem_bytes,
                              std::size_t threads_hint = 0) noexcept;

} // namespace ag::hw
