#pragma once
#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <exception>
#include <functional>
#include <memory>
#include <mutex>
#include <thread>
#include <type_traits>
#include <vector>
#include <cstdlib>

// Fixed pool.hpp
// - Exception path no longer underflows inflight_ (subtracts cleared queue size; no zeroing)
// - wait() wakes on exception OR completion, then rethrows once
// - submit_range() lazily initializes the pool (defensive)
// - Keeps TLS worker id and nesting guard semantics
//
// Public wrappers provided:
//   init_pool(n), shutdown_pool(), pool_size(), thread_index(), max_threads(),
//   submit_range(begin, end, fn), wait_for_all()

#include "ag/parallel/config.hpp" // expects: get_max_threads(), serial_override(), nesting_flag(), deterministic()

namespace ag::parallel {

// Stable TLS worker id for per-thread scratch / indexing.
inline thread_local std::size_t tls_worker_id = static_cast<std::size_t>(-1);

namespace detail {

struct RangeTask {
  // fn(begin, end); inclusive-exclusive
  std::function<void(std::size_t, std::size_t)> fn;
  std::size_t begin{0}, end{0};
};

class ThreadPool {
 public:
  ThreadPool() = default;
  ~ThreadPool() { shutdown(); }

  void init(std::size_t nthreads = 0) {
    std::lock_guard<std::mutex> lk(mu_);
    if (!workers_.empty()) return;
    std::size_t hw = std::thread::hardware_concurrency();
    if (nthreads == 0) {
      if (std::size_t cap = get_max_threads()) hw = std::min(hw ? hw : 1, cap);
    } else {
      hw = nthreads;
    }
    if (const char* e = std::getenv("AG_NUM_THREADS")) {
      long v = std::strtol(e, nullptr, 10);
      if (v > 0) hw = static_cast<std::size_t>(v);
    }
    if (hw == 0) hw = 1;
    stop_.store(false, std::memory_order_relaxed);
    inflight_.store(0, std::memory_order_relaxed);
    workers_.reserve(hw);
    for (std::size_t i = 0; i < hw; ++i) {
      workers_.emplace_back([this, i]{
        ag::parallel::tls_worker_id = i;
        this->worker_loop();
        ag::parallel::tls_worker_id = static_cast<std::size_t>(-1);
      });
    }
  }

  void shutdown() {
    {
      std::lock_guard<std::mutex> lk(mu_);
      if (workers_.empty()) return;
      stop_.store(true, std::memory_order_relaxed);
      cv_.notify_all();
    }
    for (auto &t : workers_) if (t.joinable()) t.join();
    workers_.clear();
    // Clear queue
    {
      std::lock_guard<std::mutex> lk(mu_);
      tasks_.clear();
    }
    inflight_.store(0, std::memory_order_relaxed);
    first_exc_ = nullptr;
  }

  std::size_t size() const {
    return workers_.size();
  }

  // Enqueue a task range; increases inflight counter.
  void submit(std::size_t begin, std::size_t end,
              std::function<void(std::size_t,std::size_t)> fn) {
    {
      std::lock_guard<std::mutex> lk(mu_);
      tasks_.push_back(RangeTask{std::move(fn), begin, end});
      inflight_.fetch_add(1, std::memory_order_relaxed);
    }
    cv_.notify_one();
  }

  // Wait for all tasks submitted so far, or for first exception.
  void wait() {
    std::unique_lock<std::mutex> lk(mu_);
    done_cv_.wait(lk, [this]{
      return inflight_.load(std::memory_order_relaxed) == 0 || first_exc_;
    });
    if (first_exc_) {
      auto ex = first_exc_;
      first_exc_ = nullptr;
      lk.unlock();
      std::rethrow_exception(ex);
    }
  }

  // Internal: called by workers when a task finishes.
  void complete_one() {
    std::lock_guard<std::mutex> lk(mu_);
    if (inflight_.fetch_sub(1, std::memory_order_relaxed) == 1) {
      done_cv_.notify_all();
    }
  }

  // Capture first thrown exception and notify waiters.
  void capture_exception(std::exception_ptr eptr) {
    std::lock_guard<std::mutex> lk(mu_);
    if (!first_exc_) first_exc_ = std::move(eptr);
    // Drain queued tasks so workers can exit quickly. Keep accounting correct.
    const std::size_t q = tasks_.size();
    tasks_.clear();
    if (q) inflight_.fetch_sub(q, std::memory_order_relaxed);
    // Wake both waiters and workers (so they notice empty queue / stop promptly).
    done_cv_.notify_all();
    cv_.notify_all();
  }

 private:
  void worker_loop() noexcept {
    for (;;) {
      RangeTask task;
      {
        std::unique_lock<std::mutex> lk(mu_);
        cv_.wait(lk, [this]{ return stop_.load(std::memory_order_relaxed) || !tasks_.empty(); });
        if (stop_.load(std::memory_order_relaxed) && tasks_.empty()) return;
        task = std::move(tasks_.front());
        tasks_.pop_front();
      }
      try {
        // Mark we're in a parallel region to prevent nested parallelism
        struct Nest {
          Nest() { nesting_flag() = true; }
          ~Nest(){ nesting_flag() = false; }
        } nest_guard;
        task.fn(task.begin, task.end);
      } catch (...) {
        capture_exception(std::current_exception());
      }
      complete_one();
    }
  }

  mutable std::mutex mu_;
  std::condition_variable cv_;
  std::condition_variable done_cv_;
  std::deque<RangeTask> tasks_;
  std::vector<std::thread> workers_;
  std::atomic<bool> stop_{true};
  std::atomic<std::size_t> inflight_{0};
  std::exception_ptr first_exc_{nullptr};
};

inline ThreadPool& pool() {
  static ThreadPool p;
  return p;
}

} // namespace detail

// Public API

inline void init_pool(std::size_t n_threads = 0) {
  detail::pool().init(n_threads);
}

inline void shutdown_pool() {
  detail::pool().shutdown();
}

inline std::size_t pool_size() {
  return detail::pool().size();
}

// TLS-based helpers for kernels -----------------------------------
inline std::size_t thread_index() noexcept {
  return (tls_worker_id == static_cast<std::size_t>(-1)) ? 0 : tls_worker_id;
}

inline std::size_t max_threads() noexcept {
  return pool_size();
}

// Submit a range task. Prefer using parallel_for for high-level loops.
inline void submit_range(std::size_t begin, std::size_t end,
                         const std::function<void(std::size_t,std::size_t)>& fn) {
  // Lazy init ensures no accidental "submit before init" hang.
  if (pool_size() == 0) detail::pool().init();
  detail::pool().submit(begin, end, fn);
}

inline void wait_for_all() {
  detail::pool().wait();
}

} // namespace ag::parallel
