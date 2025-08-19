#pragma once
#include <functional>
#include <memory>
#include <vector>

namespace ag {

/** Internal node for reverse-mode autograd (scalar). */
struct Node {
  double value{0.0};                       // forward value
  double grad{0.0};                        // accumulated d(Output)/d(this)
  bool   requires_grad{true};
  std::function<void()> backward;          // pushes grads to parents
  std::vector<std::shared_ptr<Node>> parents;
};

/** User-facing scalar Variable handle. */
class Variable {
public:
  Variable();                               // defaults to 0.0, requires_grad=true
  explicit Variable(double v, bool req=true);

  // accessors
  double value() const;
  double grad()  const;

  // graph ops
  void zero_grad();                         // zero this subgraph's grads
  void backward(double seed = 1.0);         // reverse pass with seed (default 1)

  // helpers for constants/flags
  static Variable constant(double v) { return Variable(v, /*req=*/false); }
  bool requires_grad() const;

  // arithmetic (friend free functions + operator sugar)
  friend Variable add(const Variable& a, const Variable& b);
  friend Variable sub(const Variable& a, const Variable& b);
  friend Variable mul(const Variable& a, const Variable& b);
  friend Variable div(const Variable& a, const Variable& b);
  friend Variable pow(const Variable& a, const Variable& b);  // a^b (double^double)
  friend Variable sinv(const Variable& x);
  friend Variable cosv(const Variable& x);
  friend Variable expv(const Variable& x);
  friend Variable neg(const Variable& x);

  // operator sugar (calls the friends above)
  friend Variable operator+(const Variable& a, const Variable& b) { return add(a,b); }
  friend Variable operator-(const Variable& a, const Variable& b) { return sub(a,b); }
  friend Variable operator*(const Variable& a, const Variable& b) { return mul(a,b); }
  friend Variable operator/(const Variable& a, const Variable& b) { return div(a,b); }
  friend Variable operator-(const Variable& x)                    { return neg(x);    }

  // overloads with scalars (promote to constants)
  friend Variable operator+(const Variable& a, double b) { return add(a, constant(b)); }
  friend Variable operator-(const Variable& a, double b) { return sub(a, constant(b)); }
  friend Variable operator*(const Variable& a, double b) { return mul(a, constant(b)); }
  friend Variable operator/(const Variable& a, double b) { return div(a, constant(b)); }

  friend Variable operator+(double a, const Variable& b) { return add(constant(a), b); }
  friend Variable operator-(double a, const Variable& b) { return sub(constant(a), b); }
  friend Variable operator*(double a, const Variable& b) { return mul(constant(a), b); }
  friend Variable operator/(double a, const Variable& b) { return div(constant(a), b); }

private:
  std::shared_ptr<Node> n;                  // private graph node
  explicit Variable(std::shared_ptr<Node> node) : n(std::move(node)) {}
};

} // namespace ag
