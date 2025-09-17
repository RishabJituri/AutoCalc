#pragma once
#include <atomic>
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <thread>
#include <algorithm>

namespace ag { namespace parallel {

// ---------- Thread cap (definition) ----------
// Global max-threads knob with sane default and ENV override.
// Tests call set_max_threads(...); callers read get_max_threads().
inline std::atomic<std::size_t>& _threads_cap() {
  static std::atomic<std::size_t> cap{[]{
    // default: clamp env AG_THREADS or hardware_concurrency() to at least 1
    auto env = std::getenv("AG_THREADS");
    std::size_t hw = std::max<std::size_t>(1, std::thread::hardware_concurrency() ? std::thread::hardware_concurrency() : 4);
    if (!env) return hw;
    // parse env
    std::size_t v = 0;
    for (const char* p = env; *p; ++p) {
      if (*p < '0' || *p > '9') { v = hw; break; }
      v = v * 10 + std::size_t(*p - '0');
    }
    if (v == 0) v = 1;
    return v;
  }()};
  return cap;
}
inline void set_max_threads(std::size_t n) {
  if (n == 0) n = 1;
  _threads_cap().store(n, std::memory_order_relaxed);
}
inline std::size_t get_max_threads() {
  return _threads_cap().load(std::memory_order_relaxed);
}

// ---------- ScopedSerial (public knob; TLS depth) ----------
struct ScopedSerial {
  ScopedSerial(){ ++depth_ref(); }
  ~ScopedSerial(){ --depth_ref(); }
  ScopedSerial(const ScopedSerial&) = delete;
  ScopedSerial& operator=(const ScopedSerial&) = delete;
  static int depth(){ return depth_ref(); }
private:
  static  int& depth_ref(){ static thread_local int d = 0; return d; }
};

// ---------- Nested guard expected by pool.hpp ----------
inline bool& nesting_flag() { static thread_local bool f = false; return f; }
struct NestedParallelGuard {
  NestedParallelGuard(){ nesting_flag() = true; }
  ~NestedParallelGuard(){ nesting_flag() = false; }
  NestedParallelGuard(const NestedParallelGuard&) = delete;
  NestedParallelGuard& operator=(const NestedParallelGuard&) = delete;
  static bool active(){ return nesting_flag(); }
};

// ---------- Determinism (env + runtime) ----------
inline bool _env_bool(const char* name, bool def=false) {
  if (const char* s = std::getenv(name)) {
    if (!std::strcmp(s,"1") || !std::strcmp(s,"true") || !std::strcmp(s,"TRUE")) return true;
    if (!std::strcmp(s,"0") || !std::strcmp(s,"false")|| !std::strcmp(s,"FALSE")) return false;
  }
  return def;
}
inline std::atomic<bool>& deterministic_flag() {
  static std::atomic<bool> v{ _env_bool("AG_DETERMINISTIC", false) };
  return v;
}
inline void set_deterministic(bool on) {
  deterministic_flag().store(on, std::memory_order_relaxed);
}
inline bool deterministic_enabled() {
  const bool env_now = _env_bool("AG_DETERMINISTIC", deterministic_flag().load(std::memory_order_relaxed));
  if (env_now != deterministic_flag().load(std::memory_order_relaxed)) {
    deterministic_flag().store(env_now, std::memory_order_relaxed);
  }
  return env_now;
}

// ---------- Deterministic *parallel* permission (env + scoped) ----------
// Global permission (default: off, preserves legacy behavior)
//   Env: AG_DETERMINISTIC_PARALLEL=0|1
inline std::atomic<bool>& deterministic_parallel_flag() {
  static std::atomic<bool> v{ _env_bool("AG_DETERMINISTIC_PARALLEL", false) };
  return v;
}
inline void set_deterministic_parallel_global(bool on) {
  deterministic_parallel_flag().store(on, std::memory_order_relaxed);
}
inline bool deterministic_parallel_global() {
  const bool env_now = _env_bool("AG_DETERMINISTIC_PARALLEL",
                                 deterministic_parallel_flag().load(std::memory_order_relaxed));
  if (env_now != deterministic_parallel_flag().load(std::memory_order_relaxed)) {
    deterministic_parallel_flag().store(env_now, std::memory_order_relaxed);
  }
  return env_now;
}

// Scoped permission (thread-local depth counter). Use this around ops that are
// deterministic-by-construction (e.g., tiled GEMM with disjoint-output tiles).
inline int& _detpar_scope_depth() { static thread_local int d = 0; return d; }

struct ScopedDeterministicParallel {
  ScopedDeterministicParallel() { ++_detpar_scope_depth(); }
  ~ScopedDeterministicParallel() { --_detpar_scope_depth(); }
  ScopedDeterministicParallel(const ScopedDeterministicParallel&) = delete;
  ScopedDeterministicParallel& operator=(const ScopedDeterministicParallel&) = delete;
  static int depth() { return _detpar_scope_depth(); }
};

inline bool deterministic_parallel_active() {
  return deterministic_parallel_global() || (_detpar_scope_depth() > 0);
}

// ---------- Unified serial gate ----------
inline bool serial_override() {
  // Treat max_threads==1 as serial; parallel_for also checks T==1.
  const bool max1 = get_max_threads() <= 1;
  return ScopedSerial::depth() > 0
      || nesting_flag()
      || (deterministic_enabled() && !deterministic_parallel_active())
      || max1;
}

}} // namespace ag::parallel
