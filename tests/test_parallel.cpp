#include "test_framework.hpp"
#include "ag/parallel/parallel_for.hpp"
#include "ag/parallel/config.hpp"
#include "ag/parallel/transform.hpp"
#include "ag/parallel/pool.hpp"

#include <vector>
#include <atomic>
#include <stdexcept>
#include <cstdlib>

using ag::parallel::parallel_for;
// using ag::parallel::parallel_for_2d;
using ag::parallel::transform;
using ag::parallel::zip_transform;
using ag::parallel::reduce;
using ag::parallel::for_rows;
using ag::parallel::set_max_threads;
using ag::parallel::set_deterministic;
using ag::parallel::ScopedSerial;

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

// ----------------- new tests below -----------------

TEST("parallel/scoped_serial_forces_single_call") {
  const std::size_t n = 20'000;
  set_max_threads(8);
  std::size_t calls = 0;

  {
    ScopedSerial s; // force serial in this scope
    parallel_for(n, /*grain=*/64, [&](std::size_t i0, std::size_t i1){
      ++calls;
      // do some work
      for (std::size_t i = i0; i < i1; ++i) {}
    });
  }

  ASSERT_TRUE(calls == 1);
}

TEST("parallel/deterministic_mode_forces_serial_globally") {
  const std::size_t n = 50'000;
  set_max_threads(8);
  set_deterministic(true);

  std::size_t calls = 0;
  parallel_for(n, /*grain=*/64, [&](std::size_t i0, std::size_t i1){
    ++calls;
    for (std::size_t i = i0; i < i1; ++i) {}
  });

  // turn it back off so we don't affect other tests
  set_deterministic(false);
  ASSERT_TRUE(calls == 1);
}

TEST("parallel/auto_grain_overload_uses_multiple_chunks") {
  const std::size_t n = 100'000;
  set_max_threads(4); // expect 4 chunks with the ~8x heuristic
  std::size_t chunks = 0;

  parallel_for(n, [&](std::size_t i0, std::size_t i1){
    (void)i0; (void)i1;
    ++chunks;
  });

  ASSERT_TRUE(chunks == 4);
}

TEST("parallel/max_threads_one_is_serial") {
  const std::size_t n = 10'000;
  set_max_threads(1);

  std::size_t calls = 0;
  parallel_for(n, /*grain=*/64, [&](std::size_t i0, std::size_t i1){
    ++calls;
    for (std::size_t i = i0; i < i1; ++i) {}
  });

  ASSERT_TRUE(calls == 1);
}

TEST("parallel/grain_zero_is_sanitized") {
  const std::size_t n = 8'192;
  set_max_threads(4);
  std::size_t chunks = 0;

  // grain==0 should be treated as 1, so we'll definitely parallelize
  parallel_for(n, /*grain=*/0, [&](std::size_t i0, std::size_t i1){
    (void)i0; (void)i1; ++chunks;
  });

  ASSERT_TRUE(chunks == 4);
}

// TEST("parallel/2d_tiler_covers_each_cell_once") {
//   const std::size_t H = 13, W = 17;     // not multiples of tile sizes
//   const std::size_t tileH = 5, tileW = 4;
//   std::vector<int> grid(H * W, 0);
//   set_max_threads(8);

//   parallel_for_2d(H, W, tileH, tileW, [&](std::size_t h0, std::size_t h1,
//                                           std::size_t w0, std::size_t w1){
//     for (std::size_t h = h0; h < h1; ++h) {
//       for (std::size_t w = w0; w < w1; ++w) {
//         grid[h * W + w] += 1;
//       }
//     }
//   });

//   // every cell should be hit exactly once
//   ASSERT_TRUE(all_marked_once(grid));
// }

TEST("parallel/exception_propagates_after_join") {
  const std::size_t n = 10'000;
  set_max_threads(8);
  bool threw = false;

  try {
    parallel_for(n, /*grain=*/256, [&](std::size_t i0, std::size_t i1){
      // make one block throw
      if (i0 == 0) {
        throw std::runtime_error("boom");
      }
      // do some work
      for (std::size_t i = i0; i < i1; ++i) {}
    });
  } catch (const std::runtime_error& e) {
    threw = true;
    (void)e;
  }

  ASSERT_TRUE(threw);
}

TEST("transform/unary_relu_shape_and_values") {
  const std::size_t n = 10'000;
  std::vector<float> x(n), y(n, -1.f);
  for (size_t i=0;i<n;++i) x[i] = (i%5==0)? -float(i) : float(i%7);

  transform(n, [&](std::size_t i){
    float v = x[i]; y[i] = v > 0.f ? v : 0.f;
  });

  for (size_t i=0;i<n;++i) {
    float want = x[i] > 0.f ? x[i] : 0.f;
    if (y[i] != want) { ASSERT_TRUE(false); break; }
  }
}

TEST("zip_transform/elementwise_add_matches_serial") {
  const std::size_t n = 50'000;
  std::vector<float> a(n), b(n), out(n);
  for (size_t i=0;i<n;++i) { a[i] = std::sin(float(i)); b[i] = std::cos(float(i)); }

  zip_transform(n, [&](std::size_t i){ out[i] = a[i] + b[i]; });

  // verify against serial reference
  for (size_t i=0;i<n;++i) {
    float want = a[i] + b[i];
    if (std::abs(out[i] - want) > 1e-6f) { ASSERT_TRUE(false); break; }
  }
}

TEST("transform/empty_n_is_noop") {
  std::vector<int> v;
  int touched = 0;
  transform(v.size(), [&](std::size_t){ ++touched; });
  ASSERT_TRUE(touched == 0);
}


TEST("reduce/argmax_matches_serial") {
  const std::size_t n = 100'003;
  std::vector<float> x(n);
  for (size_t i=0;i<n;++i) x[i] = std::cos(0.01f*float(i)) + 0.001f*float(i%13);

  struct ArgMax { std::size_t idx; float val; };
  auto res = reduce<ArgMax>(
    n, ArgMax{0, -std::numeric_limits<float>::infinity()},
    [&](ArgMax& a, std::size_t i){ if (x[i] > a.val) { a.val = x[i]; a.idx = i; } },
    [](ArgMax A, ArgMax B){ return (A.val >= B.val) ? A : B; }
  );

  // serial reference
  ArgMax ref{0, x[0]};
  for (size_t i=1;i<n;++i) if (x[i] > ref.val) { ref.val=x[i]; ref.idx=i; }

  ASSERT_TRUE(res.idx == ref.idx && std::abs(res.val - ref.val) < 1e-6f);
}

TEST("reduce/auto_grain_overload_works") {
  const std::size_t n = 123'457;
  std::vector<int> v(n);
  for (size_t i=0;i<n;++i) v[i] = int(i%7) - 3;
  int s = reduce<int>(
    n, 0,
    [&](int& acc, std::size_t i){ acc += v[i]; },
    [](int a, int b){ return a + b; }
  );
  // serial
  int ref = 0; for (auto z : v) ref += z;
  ASSERT_TRUE(s == ref);
}

// ---------- for_rows ----------
TEST("for_rows/covers_each_row_once") {
  const std::size_t rows=257, cols=37;
  std::vector<int> row_hits(rows, 0);
  for_rows(rows, [&](std::size_t r){
    ++row_hits[r];
    // fake row loop
    for (std::size_t c=0;c<cols;++c) {}
  });
  ASSERT_TRUE(all_marked_once(row_hits));
}


