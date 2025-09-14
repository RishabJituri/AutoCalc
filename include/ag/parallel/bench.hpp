#pragma once
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <thread>
#include <atomic>

namespace ag::parallel {

inline bool bench_enabled_once() {
#if defined(AG_PAR_BENCH) && AG_PAR_BENCH
  return true;
#else
  static std::atomic<int> cached{-1};
  int v = cached.load(std::memory_order_relaxed);
  if (v < 0) {
    const char* e = std::getenv("AG_BENCH");
    v = (e && *e && *e != '0') ? 1 : 0;
    cached.store(v, std::memory_order_relaxed);
  }
  return v == 1;
#endif
}

struct ScopedTimer {
  const char* label;
  bool on;
  std::chrono::steady_clock::time_point t0;

  explicit ScopedTimer(const char* lbl)
    : label(lbl), on(bench_enabled_once()),
      t0(std::chrono::steady_clock::now()) {}

  ~ScopedTimer() {
    if (!on) return;
    using namespace std::chrono;
    auto ms = duration_cast<milliseconds>(steady_clock::now() - t0).count();
    std::fprintf(stderr, "[AG_BENCH] %s | %lld ms | tid=%zu\n",
                 label ? label : "(unnamed)",
                 static_cast<long long>(ms),
                 static_cast<std::size_t>(std::hash<std::thread::id>{}(std::this_thread::get_id())));
    std::fflush(stderr);
  }
};

} // namespace ag::parallel

#define AG_CONCAT_AG_TIMER(a,b) AG_CONCAT_AG_TIMER_IMPL(a,b)
#define AG_CONCAT_AG_TIMER_IMPL(a,b) a##b
#define AG_UNIQUE_AG_TIMER AG_CONCAT_AG_TIMER(__ag_timer_, __LINE__)
#define AG_BENCH(label_literal) ::ag::parallel::ScopedTimer AG_UNIQUE_AG_TIMER{label_literal}
