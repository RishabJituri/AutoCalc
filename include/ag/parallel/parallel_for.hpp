#pragma once
#include <cstddef>
#include <thread>
#include <vector>
#include <cstdlib>
#include <atomic>
#include <algorithm>
#include <exception>
#include <string>

#include "ag/parallel/config.hpp" // determinism + flags

// Minimal parallel_for utility for AutoCalc (ag::parallel)
// - balanced partitioning, nested-parallel guard, exception-safe
// - determinism: AG_DETERMINISTIC=1 forces serial
// - AG_THREADS caps threads; override via ag::parallel::set_max_threads

namespace ag::parallel {

// Core API: partition [0, n) into blocks and call fn(i0,i1) per block
template <class Fn>
inline void parallel_for(std::size_t n, std::size_t grain, Fn&& fn) {
  if (n == 0) return;
  if (grain == 0) grain = 1;

  const bool must_serial =
      nesting_flag() || serial_override() || deterministic() ||
      (get_max_threads() <= 1) || (n <= grain);

  if (must_serial) {
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

// Convenience overload: choose a sensible grain automatically
template <class Fn>
inline void parallel_for(std::size_t n, Fn&& fn) {
  if (n == 0) return;
  // ~8 chunks per thread heuristic
  const std::size_t Tthr = std::max<std::size_t>(1, std::min(get_max_threads(), n));
  const std::size_t grain = std::max<std::size_t>(1, n / (Tthr * 8));
  parallel_for(n, grain, std::forward<Fn>(fn));
}

// 2D tiler for later linalg/conv: fn(h0,h1,w0,w1)
template <class Fn>
inline void parallel_for_2d(std::size_t H, std::size_t W,
                            std::size_t tileH, std::size_t tileW,
                            Fn&& fn) {
  if (H == 0 || W == 0) return;
  tileH = tileH ? tileH : 1;
  tileW = tileW ? tileW : 1;

  const std::size_t nTH = (H + tileH - 1) / tileH;
  const std::size_t nTW = (W + tileW - 1) / tileW;
  const std::size_t total = nTH * nTW;

  parallel_for(total, [&](std::size_t i0, std::size_t i1){
    for (std::size_t t = i0; t < i1; ++t) {
      const std::size_t th = t / nTW;
      const std::size_t tw = t % nTW;
      const std::size_t h0 = th * tileH;
      const std::size_t w0 = tw * tileW;
      const std::size_t h1 = (h0 + tileH < H) ? (h0 + tileH) : H;
      const std::size_t w1 = (w0 + tileW < W) ? (w0 + tileW) : W;
      fn(h0, h1, w0, w1);
    }
  });
}

} // namespace ag::parallel