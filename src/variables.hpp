#pragma once

#include <functional>
#include <memory>
#include <vector>

namespace ag {

// -----------------------------
// Graph Node
// -----------------------------
struct Node {
    std::vector<double> value;           // flattened tensor values
    std::vector<size_t> shape;           // tensor shape
    std::vector<double> grad;            // flattened grads (same length as value)
    bool requires_grad = false;          // track gradients?

    std::vector<std::shared_ptr<Node>> parents; // inputs to this node
    std::function<void()> backward;             // local backward closure (accumulates into parents)
};

// -----------------------------
// Variable (user-facing handle)
// -----------------------------
class Variable {
public:
    std::shared_ptr<Node> n; // shared node (enables graph reuse)

    Variable();
    explicit Variable(std::shared_ptr<Node> node); // wrap existing node (preserve graph & backward)
    Variable(const std::vector<double>& value,
             const std::vector<size_t>& shape,
             bool requires_grad = true);

    const std::vector<double>& value() const { return n->value; }
    const std::vector<double>& grad() const { return n->grad; }
    const std::vector<size_t>& shape() const { return n->shape; }
    bool requires_grad() const { return n->requires_grad; }

    void zero_grad();

    // If output is scalar (numel==1), calling backward() with no seed uses seed=1.
    // For non-scalars, pass a seed gradient matching the output size (flattened).
    void backward();
    void backward(const std::vector<double>& seed);
};

// -----------------------------
// Factory helpers
// -----------------------------
Variable scalar(double v, bool requires_grad = true);

// -----------------------------
// Ops
// -----------------------------
// Matrix multiplication with NumPy-style batched broadcasting on leading dims.
// A:[B..., M, K] @ B:[C..., K, N] -> [broadcast(B...,C...), M, N]
Variable matmul(const Variable& A, const Variable& B);

} // namespace ag