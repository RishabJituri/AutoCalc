#pragma once
#include <memory>
#include <vector>
#include <functional>
#include <cstddef>
#include <atomic>

namespace ag {

// Global grad mode (thread-local). Controls whether new nodes require_grad.
inline thread_local bool __grad_enabled = true;
inline bool is_grad_enabled() { return __grad_enabled; }
inline void set_grad_enabled(bool v) { __grad_enabled = v; }

// ── Live‑Node counter (leak detection) ──────────────────────────────────────
// Incremented in Node ctor, decremented in Node dtor.
// Exposed to Python via `ag.live_node_count()`.
inline std::atomic<int64_t> g_live_node_count{0};

struct Node {
  std::vector<float> value;                // flattened tensor
  std::vector<float> grad;                 // same size as value
  std::vector<std::size_t> shape;           // tensor shape
  bool requires_grad = false;

  std::vector<std::shared_ptr<Node>> parents; // inputs
  std::function<void()> backward;             // local VJP

  Node()  { g_live_node_count.fetch_add(1, std::memory_order_relaxed); }
  ~Node() { g_live_node_count.fetch_sub(1, std::memory_order_relaxed); }

  // Non-copyable, non-movable (shared_ptr manages lifetime)
  Node(const Node&) = delete;
  Node& operator=(const Node&) = delete;
  Node(Node&&) = delete;
  Node& operator=(Node&&) = delete;
};

class Variable {
public:
  Variable();                                                   // empty node
  explicit Variable(std::shared_ptr<Node> node);                // wrap existing
  Variable(const std::vector<float>& value,
           const std::vector<std::size_t>& shape,
           bool requires_grad = true);

  const std::vector<float>& value() const;
  const std::vector<float>& grad()  const;
  const std::vector<std::size_t>& shape() const;
  bool requires_grad() const;

  void zero_grad();                                // zero across reachable subgraph
  void backward();                                 // scalar output → seed 1
  void backward(const std::vector<float>& seed);  // tensor output → explicit seed

  // expose node handle for ops implementation
  std::shared_ptr<Node> n;

private:
  friend Variable make_from_node(std::shared_ptr<Node>);
};

Variable make_from_node(std::shared_ptr<Node> node);

// ===== Ops (declarations) =====
// Elementwise (broadcasting)
Variable add(const Variable& a, const Variable& b);
Variable sub(const Variable& a, const Variable& b);
Variable mul(const Variable& a, const Variable& b);
Variable div(const Variable& a, const Variable& b);
Variable neg(const Variable& x);
Variable sinv(const Variable& x);
Variable cosv(const Variable& x);
Variable expv(const Variable& x);
Variable pow(const Variable& base, const Variable& exponent);

// Linear algebra
// Batched matmul: A:[B..., M, K] @ B:[C..., K, N] -> [broadcast(B...,C...), M, N]
Variable matmul(const Variable& A, const Variable& B);

// Materialized last-two-dimension transpose: returns a new Variable with the
// last two dimensions swapped and the underlying data copied/reshaped.
Variable transpose(const Variable& A);

// Materialized slice: copy a contiguous block specified by per-dim begin/end (end exclusive).
// begin and end must have the same rank as the tensor shape. Slicing currently does not
// support steps or negative indices; integer indexing is represented by begin[i]==idx and end[i]==idx+1.
Variable at(const Variable& A, const std::vector<std::size_t>& begin, const std::vector<std::size_t>& end);

} // namespace ag
