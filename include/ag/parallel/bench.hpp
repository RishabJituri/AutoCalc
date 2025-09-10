#pragma once
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <thread>

namespace ag::parallel {

inline bool bench_enabled() {
#if defined(AG_PAR_BENCH) && AG_PAR_BENCH
  return true;
#else
  static int cached = -1;
  if (cached < 0) {
    const char* e = std::getenv("AG_BENCH");
    cached = (e && *e && *e != '0') ? 1 : 0;
  }
  return cached == 1;
#endif
}

struct ScopedTimer {
  const char* label;
  std::chrono::steady_clock::time_point t0;
  explicit ScopedTimer(const char* lbl) : label(lbl), t0(std::chrono::steady_clock::now()) {}
  ~ScopedTimer() {
    using namespace std::chrono;
    auto ms = duration_cast<milliseconds>(steady_clock::now() - t0).count();
    if (bench_enabled()) {
      std::fprintf(stderr, "[AG_BENCH] %s | %lld ms | tid=%zu\n",
                   label ? label : "(unnamed)",
                   static_cast<long long>(ms),
                   static_cast<std::size_t>(std::hash<std::thread::id>{}(std::this_thread::get_id())));
      std::fflush(stderr);
    }
  }
};

} // namespace ag::parallel

// Convenience macros (no-ops unless AG_BENCH=1 or AG_PAR_BENCH is defined)
#define AG_BENCH(label_literal) ::ag::parallel::ScopedTimer AG_UNIQUE_AG_TIMER{label_literal}
#define AG_UNIQUE_AG_TIMER AG_CONCAT_AG_TIMER(__ag_timer_, __LINE__)
#define AG_CONCAT_AG_TIMER(a,b) AG_CONCAT_AG_TIMER_IMPL(a,b)
#define AG_CONCAT_AG_TIMER_IMPL(a,b) a##b
