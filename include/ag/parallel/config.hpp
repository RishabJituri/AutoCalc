#pragma once
#include <cstddef>
#include <cstdlib>
#include <thread>
#include <atomic>

namespace ag::parallel {

// ---------- Thread cap (env: AG_THREADS, or override via API) ----------
inline std::size_t& max_threads_override() {
  static std::size_t v = 0; // 0 => auto
  return v;
}
inline void set_max_threads(std::size_t t) { max_threads_override() = t; }

inline std::size_t get_max_threads() {
  std::size_t t = max_threads_override();
  if (t) return t;
  if (const char* env = std::getenv("AG_THREADS")) {
    try {
      std::size_t from_env = static_cast<std::size_t>(std::stoul(env));
      if (from_env) return from_env;
    } catch (...) {}
  }
  unsigned hw = std::thread::hardware_concurrency();
  return hw ? static_cast<std::size_t>(hw) : static_cast<std::size_t>(4);
}

// ---------- Determinism (env: AG_DETERMINISTIC=1) ----------
inline std::atomic<int> g_deterministic_cached{-1}; // -1 unknown, 0 off, 1 on

inline bool deterministic() {
#if defined(AG_PAR_DETERMINISTIC) && AG_PAR_DETERMINISTIC
  return true;
#else
  int c = g_deterministic_cached.load(std::memory_order_relaxed);
  if (c < 0) {
    const char* e = std::getenv("AG_DETERMINISTIC");
    c = (e && *e && *e != '0') ? 1 : 0;
    g_deterministic_cached.store(c, std::memory_order_relaxed);
  }
  return c == 1;
#endif
}

inline void set_deterministic(bool on) {
#if defined(AG_PAR_DETERMINISTIC) && AG_PAR_DETERMINISTIC
  (void)on;
#else
  g_deterministic_cached.store(on ? 1 : 0, std::memory_order_relaxed);
#endif
}

// ---------- Scoped serial override (force single-thread within a scope) ----------
inline int& serial_depth_flag() {
  static thread_local int depth = 0;
  return depth;
}
struct ScopedSerial {
  ScopedSerial() { ++serial_depth_flag(); }
  ~ScopedSerial() { --serial_depth_flag(); }
};
inline bool serial_override() { return serial_depth_flag() > 0; }

// ---------- Nested-parallelism guard ----------
inline bool& nesting_flag() {
  static thread_local bool v = false;
  return v;
}

} // namespace ag::parallel
