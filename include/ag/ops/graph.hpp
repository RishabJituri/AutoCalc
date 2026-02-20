#pragma once
#include "ag/core/variables.hpp"

namespace ag {
// Returns a detached copy: same values/shape, no parents, requires_grad=false.
Variable stop_gradient(const Variable& x);

// Alias: familiar PyTorch-style helper
Variable detach(const Variable& x);

// RAII guard to disable/enable grad building in current thread.
// Reentrant: nested guards correctly restore the previous state.
struct NoGradGuard {
  NoGradGuard();
  ~NoGradGuard();
  NoGradGuard(const NoGradGuard&) = delete;
  NoGradGuard& operator=(const NoGradGuard&) = delete;
private:
  bool prev_;  // saved grad-enabled state
};
}
