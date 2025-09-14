
// tests/test_pool.cpp
#include "test_framework.hpp"
#include "ag/parallel/pool.hpp"
#include "ag/parallel/config.hpp"
#include <atomic>
#include <vector>
#include <thread>
#include <mutex>
#include <unordered_set>
#include <stdexcept>

using namespace ag::parallel;

TEST("pool/basic_parallel_sum_is_correct_and_race_free") {
    set_max_threads(8);
    const std::size_t N = 100'000;
    std::vector<int> x(N, 1);
    std::atomic<long long> acc{0};

    const std::size_t stripes = 64;
    for (std::size_t s = 0; s < stripes; ++s) {
        const std::size_t a = (N * s) / stripes;
        const std::size_t b = (N * (s + 1)) / stripes;
        submit_range(a, b, [&](std::size_t i0, std::size_t i1){
            long long local = 0;
            for (std::size_t i = i0; i < i1; ++i) local += x[i];
            acc.fetch_add(local, std::memory_order_relaxed);
        });
    }
    wait_for_all();
    ASSERT_TRUE(acc.load() == static_cast<long long>(N));
}

TEST("pool/exception_rethrows_and_pool_recovers") {
    // One task throws; wait_for_all should rethrow; pool remains usable afterwards.
    std::atomic<int> ran{0};

    // First wave: one throws
    submit_range(0, 4, [&](std::size_t, std::size_t){ ++ran; });
    submit_range(0, 1, [&](std::size_t, std::size_t){ throw std::runtime_error("boom"); });
    bool threw = false;
    try { wait_for_all(); } catch (...) { threw = true; }
    ASSERT_TRUE(threw);

    // Second wave: should still work
    std::atomic<int> again{0};
    for (int k = 0; k < 8; ++k) {
        submit_range(k, k+1, [&](std::size_t, std::size_t){ ++again; });
    }
    wait_for_all();
    ASSERT_TRUE(again.load() == 8);
}

TEST("pool/lazy_init_submit_before_init_does_not_hang") {
    // submit_range lazily inits the pool if needed; must not hang.
    std::atomic<int> c{0};
    submit_range(0, 10, [&](std::size_t, std::size_t){ ++c; });
    wait_for_all();
    ASSERT_TRUE(c.load() == 1); // single task covering [0,10); function called once
}

TEST("pool/respects_thread_cap_and_thread_index") {
    // Ensure a fresh pool configured to the desired cap
    shutdown_pool();
    set_max_threads(4);
    init_pool(); // explicit init to honor cap
    std::mutex mu;
    std::unordered_set<std::size_t> tids;

    // Schedule at least as many disjoint tasks as threads
    for (int k = 0; k < 16; ++k) {
        submit_range(k, k+1, [&](std::size_t, std::size_t){
            std::lock_guard<std::mutex> lk(mu);
            tids.insert(thread_index()); // tls-based worker index (0..T-1)
        });
    }
    wait_for_all();

    const std::size_t T = pool_size();
    ASSERT_TRUE(!tids.empty());
    ASSERT_TRUE(tids.size() <= T);
}

TEST("pool/wait_without_work_returns_immediately") {
    wait_for_all();
    ASSERT_TRUE(true);
}

TEST("pool/many_small_tasks_cover_all_indices") {
    set_max_threads(4);
    init_pool();
    const std::size_t N = 8192;
    std::vector<std::atomic<uint8_t>> seen(N);
    for (auto& v : seen) v.store(0, std::memory_order_relaxed);

    for (std::size_t i = 0; i < N; ++i) {
        submit_range(i, i+1, [&](std::size_t a, std::size_t b){
            for (std::size_t j = a; j < b; ++j) seen[j].store(1, std::memory_order_relaxed);
        });
    }
    wait_for_all();
    for (std::size_t i = 0; i < N; ++i) ASSERT_TRUE(seen[i].load() == 1);
}
