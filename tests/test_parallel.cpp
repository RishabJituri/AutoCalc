#include "test_framework.hpp"
#include "ag/core/parallel.hpp"  // adjust to "ag/core/parallel.hpp" if you move it under core/
#include <vector>
#include <atomic>
#include <cstdlib>

using ag::parallel_for;
using ag::set_max_threads;

// Helper: verify that every index in [0,n) is touched exactly once
static bool all_marked_once(const std::vector<int>& marks) {
  for (size_t i = 0; i < marks.size(); ++i)
    if (marks[i] != 1) return false;
  return true;
}

TEST("parallel/basic_cover_no_overlap") {
  const std::size_t n = 10000;
  set_max_threads(8);
  std::vector<int> marks(n, 0);

  parallel_for(n, /*grain=*/128, [&](std::size_t i0, std::size_t i1){
    for (std::size_t i = i0; i < i1; ++i) {
      // Each i is unique to a single thread's sub-interval
      marks[i] += 1;
    }
  });

  ASSERT_TRUE(all_marked_once(marks));
}

TEST("parallel/single_thread_when_small") {
  const std::size_t n = 100;
  set_max_threads(8);
  std::size_t calls = 0;
  std::size_t got0=999, got1=999;

  parallel_for(n, /*grain=*/1000, [&](std::size_t i0, std::size_t i1){
    ++calls; got0 = i0; got1 = i1;
    for (std::size_t i = i0; i < i1; ++i) {}
  });

  ASSERT_TRUE(calls == 1);
  ASSERT_TRUE(got0 == 0 && got1 == n);
}

TEST("parallel/respects_max_threads_via_grain") {
  const std::size_t n = 1000;
  set_max_threads(3); // cap threads at 3
  std::size_t chunks = 0;

  parallel_for(n, /*grain=*/100, [&](std::size_t i0, std::size_t i1){
    (void)i0; (void)i1;
    ++chunks;
  });

  // ceil(1000/100)=10, min(3, 10) = 3
  ASSERT_TRUE(chunks == 3);
}

TEST("parallel/nested_stays_single_thread_inside") {
  const std::size_t n = 4096;
  set_max_threads(8);
  std::vector<int> marks(n, 0);

  parallel_for(n, /*grain=*/256, [&](std::size_t i0, std::size_t i1){
    // Inner parallel_for should run single-threaded due to nesting guard
    parallel_for(i1 - i0, /*grain=*/64, [&](std::size_t j0, std::size_t j1){
      for (std::size_t j = j0; j < j1; ++j) {
        marks[i0 + j] += 1;
      }
    });
  });

  ASSERT_TRUE(all_marked_once(marks));
}
