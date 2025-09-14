#pragma once
#include <cstddef>
#include <vector>
#include <atomic>
#include "ag/parallel/parallel_for.hpp"

namespace ag::parallel {

// unary map: body(i)
template<class Fn>
inline void transform(std::size_t n, std::size_t grain, Fn&& body) {
  if (!n) return;
  ag::parallel::parallel_for(n, grain, [&](std::size_t i0, std::size_t i1){
    #if defined(__clang__) || defined(__GNUC__)
    #pragma clang loop vectorize(enable) interleave(enable)
    #endif
    for (std::size_t i = i0; i < i1; ++i) body(i);
  });
}

template<class Fn>
inline void transform(std::size_t n, Fn&& body) {
  if (!n) return;
  ag::parallel::parallel_for(n, [&](std::size_t i0, std::size_t i1){
    #if defined(__clang__) || defined(__GNUC__)
    #pragma clang loop vectorize(enable) interleave(enable)
    #endif
    for (std::size_t i = i0; i < i1; ++i) body(i);
  });
}

// binary map: body(i)
template<class Fn>
inline void zip_transform(std::size_t n, std::size_t grain, Fn&& body) {
  if (!n) return;
  ag::parallel::parallel_for(n, grain, [&](std::size_t i0, std::size_t i1){
    #if defined(__clang__) || defined(__GNUC__)
    #pragma clang loop vectorize(enable) interleave(enable)
    #endif
    for (std::size_t i = i0; i < i1; ++i) body(i);
  });
}

template<class Fn>
inline void zip_transform(std::size_t n, Fn&& body) {
  if (!n) return;
  ag::parallel::parallel_for(n, [&](std::size_t i0, std::size_t i1){
    #if defined(__clang__) || defined(__GNUC__)
    #pragma clang loop vectorize(enable) interleave(enable)
    #endif
    for (std::size_t i = i0; i < i1; ++i) body(i);
  });
}

// rows launcher: body(row)
template<class Fn>
inline void for_rows(std::size_t rows, std::size_t grain, Fn&& body) {
  if (!rows) return;
  ag::parallel::parallel_for(rows, grain, [&](std::size_t r0, std::size_t r1){
    for (std::size_t r = r0; r < r1; ++r) body(r);
  });
}

template<class Fn>
inline void for_rows(std::size_t rows, Fn&& body) {
  if (!rows) return;
  ag::parallel::parallel_for(rows, [&](std::size_t r0, std::size_t r1){
    for (std::size_t r = r0; r < r1; ++r) body(r);
  });
}

template <class T, class Apply, class Combine>
inline T reduce(std::size_t n, std::size_t grain,
                T init, Apply&& apply, Combine&& combine) {
  if (n == 0) return init;

  // decide number of blocks similar to parallel_for
  const std::size_t cap    = std::max<std::size_t>(1, get_max_threads());
  const std::size_t blocks = std::max<std::size_t>(
      std::min<std::size_t>(cap, (n + grain - 1) / grain), std::size_t{1});

  std::vector<T> partial(blocks, init);
  std::atomic<std::size_t> slot{0};

  ag::parallel::parallel_for(n, grain, [&](std::size_t i0, std::size_t i1){
    const std::size_t s = slot.fetch_add(1, std::memory_order_relaxed); // unique slot
    T acc = init;
    for (std::size_t i = i0; i < i1; ++i) apply(acc, i);
    partial[s] = std::move(acc);
  });

  // fixed left-to-right combine for stability
  const std::size_t used = std::min<std::size_t>(slot.load(std::memory_order_relaxed), blocks);
  T out = init;
  for (std::size_t k = 0; k < used; ++k) out = combine(out, partial[k]);
  return out;
}

template <class T, class Apply, class Combine>
inline T reduce(std::size_t n, T init, Apply&& apply, Combine&& combine) {
  if (n == 0) return init;
  const std::size_t Tthr  = std::max<std::size_t>(1, std::min(get_max_threads(), n));
  const std::size_t grain = std::max<std::size_t>(1, n / (Tthr * 8)); // ~8 blocks/thread
  return reduce(n, grain, std::move(init),
                std::forward<Apply>(apply),
                std::forward<Combine>(combine));
}

} // namespace ag::parallel