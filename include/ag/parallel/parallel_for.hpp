#pragma once
// Pool-backed parallel_for (lazy persistent pool, grain-aware, nested-safe, deterministic)
#include <cstddef>
#include <algorithm>
#include <atomic>
#include <exception>
#include <utility>
#include <vector>

#include "ag/parallel/config.hpp"
#include "ag/parallel/pool.hpp"

namespace ag { namespace parallel {

// 1-D parallel_for
template <class Fn>
inline void parallel_for(std::size_t n, std::size_t grain, Fn&& body) {
  using std::size_t;

  const size_t Tcap = ag::parallel::get_max_threads();
  const size_t T    = std::max<size_t>(1, std::min(Tcap, n));

  // Serial gates (user-forced, deterministic, or nested)
  if (T == 1 || n == 0 || ag::parallel::serial_override()) {
    body(0, n);
    return;
  }

  // Determine number of chunks
  const size_t chunks = (grain == 0)
      ? T
      : std::max<size_t>(1, std::min(T, (n + grain - 1) / grain));

  if (chunks == 1) {
    body(0, n);
    return;
  }

  auto k_block = [&](size_t k)->std::pair<size_t,size_t> {
    const size_t q = n / chunks, r = n % chunks;
    const size_t lo = k * q + (k < r ? k : r);
    return {lo, lo + q + (k < r)};
  };

  std::exception_ptr first_ep = nullptr;
  std::atomic<bool> seen_ep{false};

  for (size_t k = 0; k < chunks; ++k) {
    auto range = k_block(k);
    const std::size_t lo = range.first;
    const std::size_t hi = range.second;

    ag::parallel::submit_range(lo, hi, [&](std::size_t i0, std::size_t i1){
      ag::parallel::NestedParallelGuard _nested; // inner PF runs serial
      try {
        if (i1 > i0) body(i0, i1);
      } catch (...) {
        if (!seen_ep.exchange(true, std::memory_order_relaxed)) {
          first_ep = std::current_exception();
        }
      }
    });
  }

  ag::parallel::wait_for_all();
  if (first_ep) std::rethrow_exception(first_ep);
}

// Overload: grain defaults to 0 (auto)
template <class Fn>
inline void parallel_for(std::size_t n, Fn&& body) {
  parallel_for(n, /*grain=*/0, std::forward<Fn>(body));
}

// // 2-D tiling helper
// template <class Fn2D>
// inline void parallel_for_2d(std::size_t H, std::size_t W,
//                             std::size_t tileH, std::size_t tileW,
//                             Fn2D&& body) {
//   if (H == 0 || W == 0) return;
//   if (tileH == 0) tileH = H;
//   if (tileW == 0) tileW = W;

//   const std::size_t th = (H + tileH - 1) / tileH;
//   const std::size_t tw = (W + tileW - 1) / tileW;
//   const std::size_t total = th * tw;

//   parallel_for(total, /*grain=*/0, [&](std::size_t t0, std::size_t t1){
//     for (std::size_t t = t0; t < t1; ++t) {
//       const std::size_t ih = t / tw;
//       const std::size_t iw = t % tw;
//       const std::size_t h0 = ih * tileH;
//       const std::size_t h1 = std::min(h0 + tileH, H);
//       const std::size_t w0 = iw * tileW;
//       const std::size_t w1 = std::min(w0 + tileW, W);
//       body(h0, h1, w0, w1);
//     }
//   });
// }

}} // namespace ag::parallel
