#include "ag/sys/hw.hpp"

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <vector>

#if defined(__APPLE__)
  #include <sys/types.h>
  #include <sys/sysctl.h>
#elif defined(_WIN32)
  #define NOMINMAX
  #include <windows.h>
#else
  #include <cstdio>
  #include <unistd.h>
#endif

namespace ag::sys {
namespace {

CacheInfo g_ci;                // detected values (mutable by reload)
std::once_flag g_once;         // first-time init guard
std::mutex g_reload_mtx;       // protects re-detect in reload()

static std::size_t parse_bytes_env(const char* s, std::size_t def) {
  if (!s || !*s) return def;
  char* end = nullptr;
  double v = std::strtod(s, &end);
  if (v <= 0) return def;
  while (end && *end == ' ') ++end;
  // Accept K/M/G suffix (case-insensitive)
  if (end && *end) {
    switch (std::tolower(static_cast<unsigned char>(*end))) {
      case 'g': v *= 1024.0; [[fallthrough]];
      case 'm': v *= 1024.0; [[fallthrough]];
      case 'k': v *= 1024.0; [[fallthrough]];
      default: break;
    }
  }
  return static_cast<std::size_t>(v);
}

#if defined(__APPLE__)
static std::size_t sysctl_size(const char* key, std::size_t def) {
  std::size_t v = 0; size_t len = sizeof(v);
  if (sysctlbyname(key, &v, &len, nullptr, 0) == 0 && v) return v;
  return def;
}
#elif !defined(_WIN32)
static std::size_t read_sys_size(const char* path) {
  FILE* f = std::fopen(path, "r");
  if (!f) return 0;
  char buf[64] = {0};
  const auto n = std::fread(buf, 1, sizeof(buf) - 1, f);
  std::fclose(f);
  if (!n) return 0;
  return parse_bytes_env(buf, 0);
}
#endif

static void detect(CacheInfo& ci) {
  // Start from conservative defaults
  ci = CacheInfo{};

#if defined(__APPLE__)
  ci.line = sysctl_size("hw.cachelinesize", ci.line);
  ci.l1d  = sysctl_size("hw.l1dcachesize",  ci.l1d);
  ci.l2   = sysctl_size("hw.l2cachesize",   ci.l2);
#elif defined(_WIN32)
  DWORD len = 0;
  GetLogicalProcessorInformationEx(RelationCache, nullptr, &len);
  if (GetLastError() == ERROR_INSUFFICIENT_BUFFER && len) {
    std::vector<std::uint8_t> buf(len);
    if (GetLogicalProcessorInformationEx(
            RelationCache,
            reinterpret_cast<PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX>(buf.data()),
            &len)) {
      BYTE* p = buf.data();
      while (p < buf.data() + len) {
        auto info =
            reinterpret_cast<PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX>(p);
        if (info->Relationship == RelationCache) {
          const auto& c = info->Cache;
          if (c.Level == 1 && c.Type == CacheData)
            ci.l1d = std::max<std::size_t>(ci.l1d, c.CacheSize);
          if (c.Level == 2)
            ci.l2 = std::max<std::size_t>(ci.l2, c.CacheSize);
          ci.line = std::max<std::size_t>(ci.line, c.LineSize);
        }
        p += info->Size;
      }
    }
  }
#else
  long l1 = sysconf(_SC_LEVEL1_DCACHE_SIZE);
  long l2 = sysconf(_SC_LEVEL2_CACHE_SIZE);
  long ln = sysconf(_SC_LEVEL1_DCACHE_LINESIZE);
  if (l1 > 0) ci.l1d = static_cast<std::size_t>(l1);
  if (l2 > 0) ci.l2  = static_cast<std::size_t>(l2);
  if (ln > 0) ci.line= static_cast<std::size_t>(ln);

  // Fallback to /sys if needed
  if (ci.l1d == 0)
    ci.l1d = read_sys_size("/sys/devices/system/cpu/cpu0/cache/index0/size");
  if (ci.l2 == 0)
    ci.l2  = read_sys_size("/sys/devices/system/cpu/cpu0/cache/index2/size");
  if (ci.line == 0) {
    FILE* f = std::fopen("/sys/devices/system/cpu/cpu0/cache/index0/coherency_line_size", "r");
    if (f) { unsigned v = 0; if (std::fscanf(f, "%u", &v) == 1 && v) ci.line = v; std::fclose(f); }
  }
#endif

  // Env overrides (accept raw or with K/M/G suffix)
  if (const char* s = std::getenv("AG_CACHE_LINE")) ci.line = parse_bytes_env(s, ci.line);
  if (const char* s = std::getenv("AG_L1D"))        ci.l1d  = parse_bytes_env(s, ci.l1d);
  if (const char* s = std::getenv("AG_L2"))         ci.l2   = parse_bytes_env(s, ci.l2);

  // Final sanity
  if (ci.line == 0) ci.line = 64;
  if (ci.l1d  == 0) ci.l1d  = (32u << 10);
  if (ci.l2   == 0) ci.l2   = (512u << 10);
}

} // namespace

const CacheInfo& cache_info() noexcept {
  std::call_once(g_once, [] { detect(g_ci); });
  return g_ci;
}

void reload() noexcept {
  // Ensure we’ve run once (safe even if already initialized)
  (void)cache_info();
  // Re-run detection under a mutex (don’t try to assign once_flag)
  std::lock_guard<std::mutex> lk(g_reload_mtx);
  detect(g_ci);
}

std::size_t default_grain_for(std::size_t elem_bytes, std::size_t /*threads_hint*/) noexcept {
  const auto L1 = cache_info().l1d;
  const std::size_t b = elem_bytes ? elem_bytes : 1;
  std::size_t target = (L1 >> 2);   // ~ L1D/4
  std::size_t g = target / b;
  if (g == 0) g = 1;
  return g;
}

} // namespace ag::sys
