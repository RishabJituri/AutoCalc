// tests/test_parallel_per_thread.cpp
#include "test_framework.hpp"
#include "ag/parallel/per_thread.hpp"
#include "ag/parallel/pool.hpp"
#include "ag/parallel/config.hpp"
#include <vector>
#include <cstdint>
#include <atomic>

using namespace ag::parallel;

struct Scratch {
    std::vector<float> buf;
};

TEST("per_thread/alignment_reuse_isolation") {
    // Fresh pool with a small cap so we exercise multiple workers deterministically
    shutdown_pool();
    set_max_threads(4);
    init_pool();

    const std::size_t T = pool_size();
    std::vector<PerThreadSlot<Scratch>> slots(slots_for_threads(T));

    // One task per worker; record pointers and markers
    std::vector<void*> ptrs(T, nullptr);
    std::vector<float> marks(T, 0.0f);

    for (std::size_t k = 0; k < T; ++k) {
        submit_range(k, k+1, [&](std::size_t, std::size_t){
            const std::size_t tid = thread_index();

            // Same object on multiple accesses within the same task
            auto& a = slots[tid].v;
            auto& b = slots[tid].v;
            ASSERT_TRUE(&a == &b);

            // Initialize buffer and keep pointer (no reallocation since size stays the same)
            a.buf.resize(256);
            void* p1 = a.buf.data();
            a.buf[0] = static_cast<float>(tid + 1);
            void* p2 = a.buf.data();
            ASSERT_TRUE(p1 == p2);

            ptrs[tid]  = p1;
            marks[tid] = a.buf[0];
        });
    }
    wait_for_all();

    // 1) Distinct pointers across threads (isolation)
    for (std::size_t i = 0; i < T; ++i) {
        for (std::size_t j = i + 1; j < T; ++j) {
            if (ptrs[i] && ptrs[j]) {
                ASSERT_TRUE(ptrs[i] != ptrs[j]);
            }
        }
    }

    // 2) Marker preserved
    for (std::size_t t = 0; t < T; ++t) {
        if (ptrs[t]) {
            ASSERT_TRUE(slots[t].v.buf.size() >= 1);
            ASSERT_TRUE(slots[t].v.buf[0] == static_cast<float>(t + 1));
            ASSERT_TRUE(marks[t] == static_cast<float>(t + 1));
        }
    }

    // 3) thread_index range sanity
    for (std::size_t k = 0; k < T; ++k) {
        submit_range(k, k+1, [&](std::size_t, std::size_t){
            const std::size_t tid = thread_index();
            ASSERT_TRUE(tid < T);
            // Same-object check again inside another task
            ASSERT_TRUE(&slots[tid].v == &slots[tid].v);
        });
    }
    wait_for_all();
}
