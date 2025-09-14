// #include <cstdio>
// #include <vector>
// #include <random>
// #include <atomic>
// #include <algorithm>
// #include <chrono>
// #include "ag/parallel/pool.hpp"
// #include "ag/sys/hw.hpp"

// using namespace ag::parallel;

// int main() {
//   // hw probe
//   auto ci = ag::sys::cache_info();
//   std::printf("[hw] line=%zub L1d=%zu KiB L2=%zu KiB\n",
//               ci.line, ci.l1d >> 10, ci.l2 >> 10);

//   // init pool
//   init_pool();
//   const auto T = pool_size();
//   std::printf("[pool] threads=%zu\n", T);

//   // test 1: workers active & TLS ids set
//   std::vector<std::atomic<std::size_t>> hits(T);
//   for (auto& h : hits) h.store(0);
//   const std::size_t num_tasks = std::max<std::size_t>(T * 32, 64);
//   for (std::size_t i=0; i<num_tasks; ++i) {
//     submit_range(0,1, [&](std::size_t, std::size_t){
//       auto tid = thread_index();
//       if (tid < hits.size()) hits[tid].fetch_add(1);
//     });
//   }
//   wait_for_all();
//   std::size_t active=0, total=0;
//   for (std::size_t i=0;i<T;++i){ auto h=hits[i].load(); total+=h; if(h) ++active; }
//   std::printf("[test1] active_workers=%zu/%zu total_hits=%zu (expect %zu)\n",
//               active, T, total, num_tasks);

//   // test 2: parallel sum == serial sum
//   const std::size_t N = 1<<20;
//   std::vector<double> x(N);
//   std::mt19937_64 rng(123);
//   std::uniform_real_distribution<double> dist(0.0,1.0);
//   for (auto& v : x) v = dist(rng);

//   double serial=0.0; for (double v : x) serial += v;

//   std::vector<double> partials(T, 0.0);
//   auto t0 = std::chrono::high_resolution_clock::now();
//   const std::size_t blocks = std::max<std::size_t>(T*4, 1);
//   const std::size_t blk = (N + blocks - 1)/blocks;
//   for (std::size_t b=0;b<blocks;++b){
//     const std::size_t lo=b*blk, hi=std::min(N, lo+blk);
//     if (lo>=hi) break;
//     submit_range(lo,hi,[&](std::size_t L,std::size_t R){
//       double local=0.0; for (std::size_t i=L;i<R;++i) local+=x[i];
//       partials[thread_index()%T] += local;
//     });
//   }
//   wait_for_all();
//   double parallel=0.0; for (double v : partials) parallel += v;
//   auto t1 = std::chrono::high_resolution_clock::now();
//   std::printf("[test2] |diff|=%.3g\n", std::abs(parallel-serial));

//   auto par_ms = std::chrono::duration_cast<std::chrono::milliseconds>(t1-t0).count();
//   std::printf("[time] parallel=%lld ms (smoke only)\n", (long long)par_ms);

//   shutdown_pool();
//   return (std::abs(parallel-serial) < 1e-6) ? 0 : 1;
// }
#include <iostream>
#include <atomic>
#include "ag/parallel/config.hpp"
#include "ag/parallel/parallel_for.hpp"

int main() {
  using namespace ag::parallel;

  // 1) respects_max_threads_via_grain
  set_max_threads(3);
  std::size_t chunks = 0;
  parallel_for(1000, /*grain=*/100, [&](std::size_t, std::size_t){ ++chunks; });
  std::cout << "chunks_g100_T3=" << chunks << " (expect 3)\n";

  // 2) scoped serial
  std::size_t calls = 0;
  { ScopedSerial s;
    parallel_for(20000, /*grain=*/64, [&](std::size_t, std::size_t){ ++calls; });
  }
  std::cout << "calls_scoped_serial=" << calls << " (expect 1)\n";

  // 3) deterministic
  set_max_threads(8);
  set_deterministic(true);
  std::size_t calls_det = 0;
  parallel_for(50000, /*grain=*/64, [&](std::size_t, std::size_t){ ++calls_det; });
  set_deterministic(false);
  std::cout << "calls_deterministic=" << calls_det << " (expect 1)\n";

  // 4) auto-grain overload
  set_max_threads(4);
  std::size_t chunks_auto = 0;
  parallel_for(100000, [&](std::size_t, std::size_t){ ++chunks_auto; });
  std::cout << "chunks_auto_overload=" << chunks_auto << " (expect 4)\n";

  // 5) max_threads==1 serial
  set_max_threads(1);
  std::size_t calls_one = 0;
  parallel_for(10000, /*grain=*/64, [&](std::size_t, std::size_t){ ++calls_one; });
  std::cout << "calls_max1=" << calls_one << " (expect 1)\n";

  // 6) grain==0 sanitized -> auto -> T stripes
  set_max_threads(4);
  std::size_t chunks_g0 = 0;
  parallel_for(8192, /*grain=*/0, [&](std::size_t, std::size_t){ ++chunks_g0; });
  std::cout << "chunks_g0_T4=" << chunks_g0 << " (expect 4)\n";
}
