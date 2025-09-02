#pragma once
#include <memory>
#include <vector>
#include <functional>
#include <cstddef>

namespace ag {

// Global grad mode (thread-local). Controls whether new nodes require_grad.
inline thread_local bool __grad_enabled = true;
inline bool is_grad_enabled() { return __grad_enabled; }
inline void set_grad_enabled(bool v) { __grad_enabled = v; }


struct Node {
  std::vector<double> value;                // flattened tensor
  std::vector<double> grad;                 // same size as value
  std::vector<std::size_t> shape;           // tensor shape
  bool requires_grad = false;

  std::vector<std::shared_ptr<Node>> parents; // inputs
  std::function<void()> backward;             // local VJP
};

class Variable {
public:
  Variable();                                                   // empty node
  explicit Variable(std::shared_ptr<Node> node);                // wrap existing
  Variable(const std::vector<double>& value,
           const std::vector<std::size_t>& shape,
           bool requires_grad = true);

  const std::vector<double>& value() const;
  const std::vector<double>& grad()  const;
  const std::vector<std::size_t>& shape() const;
  bool requires_grad() const;

  void zero_grad();                                // zero across reachable subgraph
  void backward();                                 // scalar output → seed 1
  void backward(const std::vector<double>& seed);  // tensor output → explicit seed

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

} // namespace ag
