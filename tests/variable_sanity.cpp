#include <cassert>
#include <cmath>
#include <iostream>
#include <vector>
#include "variables.hpp"

static void expect_near(double a, double b, double tol = 1e-6) {
  double den = std::max(1.0, std::max(std::abs(a), std::abs(b)));
  double rel = std::abs(a - b) / den;
  if (rel > tol) {
    std::cerr << "EXPECT_NEAR failed: a=" << a << " b=" << b << " rel=" << rel << " tol=" << tol << "\n";
    std::abort();
  }
}

int main() {
  using namespace ag;

  // ---------- 1) Scalar seed behavior ----------
  {
    Variable x({2.0}, {1}, /*requires_grad=*/true);
    x.zero_grad();
    x.backward(); // scalar → seed 1
    assert(x.grad().size() == 1);
    expect_near(x.grad()[0], 1.0);
  }

  // ---------- 2) add: broadcasting + grads ----------
  {
    Variable a({1.0, 2.0, 3.0}, {3}, true);
    Variable b({10.0}, {1}, true);           // broadcast
    auto y = add(a, b);                       // [11,12,13]
    y.zero_grad();
    y.backward(std::vector<double>{1.0, 1.0, 1.0}); // d(sum)/dy
    // da = ones, db = sum(ones) = 3
    expect_near(a.grad()[0], 1.0);
    expect_near(a.grad()[1], 1.0);
    expect_near(a.grad()[2], 1.0);
    expect_near(b.grad()[0], 3.0);
  }

  // ---------- 3) sub: sign on rhs ----------
  {
    Variable a({5.0, 7.0}, {2}, true);
    Variable c({2.0, 3.0}, {2}, true);
    auto y = sub(a, c); // [3,4]
    y.zero_grad();
    y.backward(std::vector<double>{1.0, 1.0});
    // da += [1,1], dc += [-1,-1]
    expect_near(a.grad()[0], 1.0);
    expect_near(a.grad()[1], 1.0);
    expect_near(c.grad()[0], -1.0);
    expect_near(c.grad()[1], -1.0);
  }

  // ---------- 4) mul: product rule ----------
  {
    Variable a({2.0, 3.0}, {2}, true);
    Variable b({5.0, 7.0}, {2}, true);
    auto y = mul(a, b); // [10,21]
    y.zero_grad();
    y.backward(std::vector<double>{1.0, 1.0});
    // da += b, db += a
    expect_near(a.grad()[0], 5.0);
    expect_near(a.grad()[1], 7.0);
    expect_near(b.grad()[0], 2.0);
    expect_near(b.grad()[1], 3.0);
  }

  // ---------- 5) div: quotient rule ----------
  {
    Variable a({8.0, 9.0}, {2}, true);
    Variable b({2.0, 3.0}, {2}, true);
    auto y = div(a, b); // [4,3]
    y.zero_grad();
    y.backward(std::vector<double>{1.0, 1.0});
    // da += 1/b, db += -(a/b^2)
    expect_near(a.grad()[0], 1.0/2.0);
    expect_near(a.grad()[1], 1.0/3.0);
    expect_near(b.grad()[0], -8.0/(2.0*2.0));
    expect_near(b.grad()[1], -9.0/(3.0*3.0));
  }

  // ---------- 6) neg ----------
  {
    Variable x({-4.0, 5.0}, {2}, true);
    auto y = neg(x); // [4,-5]
    y.zero_grad();
    y.backward(std::vector<double>{1.0, 1.0});
    // dx += -1 * dy
    expect_near(x.grad()[0], -1.0);
    expect_near(x.grad()[1], -1.0);
  }

  // ---------- 7) sin / cos / exp ----------
  {
    Variable x({0.3, -0.7}, {2}, true);
    auto ys = sinv(x);
    auto yc = cosv(x);
    auto ye = expv(x);
    // Check grads individually with seed ones
    ys.zero_grad(); ys.backward(std::vector<double>{1.0, 1.0});
    expect_near(x.grad()[0], std::cos(0.3));
    expect_near(x.grad()[1], std::cos(-0.7));
    // reset
    x.zero_grad();
    yc.zero_grad(); yc.backward(std::vector<double>{1.0, 1.0});
    expect_near(x.grad()[0], -std::sin(0.3));
    expect_near(x.grad()[1], -std::sin(-0.7));
    x.zero_grad();
    ye.zero_grad(); ye.backward(std::vector<double>{1.0, 1.0});
    expect_near(x.grad()[0], std::exp(0.3));
    expect_near(x.grad()[1], std::exp(-0.7));
  }

  // ---------- 8) pow (elementwise) ----------
  {
    // Choose positive bases to avoid ln issues
    Variable base({2.0, 3.0}, {2}, true);
    Variable expn({3.0, 2.0}, {2}, true);
    auto y = pow(base, expn); // [8,9]
    y.zero_grad();
    y.backward(std::vector<double>{1.0, 1.0}); // d(sum)/dy
    // d/d(base) = p * base^(p-1)
    expect_near(base.grad()[0], 3.0 * std::pow(2.0, 3.0 - 1.0)); // 12
    expect_near(base.grad()[1], 2.0 * std::pow(3.0, 2.0 - 1.0)); // 6

    // d/d(exp) = y * ln(base)
    expect_near(expn.grad()[0], 8.0 * std::log(2.0));
    expect_near(expn.grad()[1], 9.0 * std::log(3.0));
  }

  // ---------- 9) matmul (2x2) ----------
  {
    // A = [[1,2],[3,4]], B = [[5,6],[7,8]]
    Variable A({1,2,3,4}, {2,2}, true);
    Variable B({5,6,7,8}, {2,2}, true);
    auto C = matmul(A, B);
    // Forward check: C = A @ B
    // C[0,0]=1*5+2*7=19; C[0,1]=1*6+2*8=22; C[1,0]=3*5+4*7=43; C[1,1]=3*6+4*8=50
    expect_near(C.value()[0], 19.0);
    expect_near(C.value()[1], 22.0);
    expect_near(C.value()[2], 43.0);
    expect_near(C.value()[3], 50.0);
    // Backward with seed ones → dA = dC @ B^T, dB = A^T @ dC
    C.zero_grad();
    C.backward(std::vector<double>{1.0,1.0,1.0,1.0});
    // dA should be [[11,15],[11,15]] (since B^T = [[5,7],[6,8]], row-sum with ones)
    expect_near(A.grad()[0], 11.0);
    expect_near(A.grad()[1], 15.0);
    expect_near(A.grad()[2], 11.0);
    expect_near(A.grad()[3], 15.0);
    // dB should be [[4,4],[6,6]] (A^T @ ones)
    expect_near(B.grad()[0], 4.0);
    expect_near(B.grad()[1], 4.0);
    expect_near(B.grad()[2], 6.0);
    expect_near(B.grad()[3], 6.0);
  }

  // ---------- 10) zero_grad + accumulation semantics ----------
  {
    Variable x({2.0}, {1}, true);
    // First backward
    x.zero_grad();
    x.backward(); // seed 1
    expect_near(x.grad()[0], 1.0);
    // Accumulate another backward with explicit seed 3 → total 4
    x.backward(std::vector<double>{3.0});
    expect_near(x.grad()[0], 4.0);
    // Now zero should clear it
    x.zero_grad();
    expect_near(x.grad()[0], 0.0);
  }

  std::cout << "[OK] all_ops_sanity passed\n";
  return 0;
}
