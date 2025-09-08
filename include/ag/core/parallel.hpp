#pragma once
#include <cstddef>
#include <thread>
#include <vector>
#include <cstdlib>
#include <atomic>
#include <algorithm>
#include <exception>
#include <string>

// Minimal parallel_for utility for AutoCalc
// - Header-only, no global state required
// - Spawns up to T threads per call (T from AG_THREADS env or hardware_concurrency)
// - Falls back to single-thread if nested or if n <= grain
// - Avoids nested parallelism via a thread_local guard
// - Exception-safe: captures first exception and rethrows on caller thread after join.
//
// Usage:
//   ag::parallel_for(N, 128, [&](std::size_t i0, std::size_t i1){
//     for (std::size_t i=i0; i<i1; ++i) { /* work on [i0, i1) */ }
//   });
//
// Notes:
// - Keep grain large enough (e.g., 64-1024) so thread launch overhead is amortized.
// - On Linux, link with -pthread (your Makefile already does this).
// - This is intentionally simple; you can swap to a true thread-pool later without changing call sites.

namespace ag {

inline std::size_t& max_threads_override() {
  static std::size_t v = 0; // 0 means "auto"
  return v;
}

inline void set_max_threads(std::size_t t) { max_threads_override() = t; }

inline std::size_t get_max_threads() {
  std::size_t t = max_threads_override();
  if (t) return t;
  // Allow environment override
  if (const char* env = std::getenv("AG_THREADS")) {
    try {
      std::size_t from_env = static_cast<std::size_t>(std::stoul(env));
      if (from_env) return from_env;
    } catch (...) {}
  }
  unsigned hw = std::thread::hardware_concurrency();
  return hw ? static_cast<std::size_t>(hw) : static_cast<std::size_t>(4);
}

inline bool& nesting_flag() {
  static thread_local bool v = false;
  return v;
}

template <class Fn>
inline void parallel_for(std::size_t n, std::size_t grain, Fn&& fn) {
  if (n == 0) return;
  if (grain == 0) grain = 1;

  // No parallel if small or nested
  if (nesting_flag() || n <= grain) {
    bool prev = nesting_flag();
    nesting_flag() = true;
    fn(0, n);
    nesting_flag() = prev;
    return;
  }

  const std::size_t T = std::max<std::size_t>(
      1, std::min(get_max_threads(), (n + grain - 1) / grain));
  if (T == 1) {
    bool prev = nesting_flag();
    nesting_flag() = true;
    fn(0, n);
    nesting_flag() = prev;
    return;
  }

  // Partition [0, n) into T blocks (balanced)
  const std::size_t base = n / T;
  const std::size_t rem  = n % T;

  std::vector<std::thread> workers;
  workers.reserve(T > 0 ? T - 1 : 0);

  std::atomic<bool> error{false};
  std::exception_ptr eptr = nullptr;

  auto launch_block = [&](std::size_t b){
    const std::size_t i0 = b * base + std::min<std::size_t>(b, rem);
    const std::size_t i1 = i0 + base + (b < rem ? 1 : 0);
    bool prev = nesting_flag();
    nesting_flag() = true;
    try {
      fn(i0, i1);
    } catch (...) {
      if (!error.exchange(true)) eptr = std::current_exception();
    }
    nesting_flag() = prev;
  };

  for (std::size_t b = 1; b < T; ++b) {
    workers.emplace_back([&, b]{ launch_block(b); });
  }
  // Current thread does block 0
  launch_block(0);

  for (auto& th : workers) th.join();

  if (eptr) std::rethrow_exception(eptr);
}

} // namespace ag
