#include <iostream>
#include <cassert>
#include "variables.hpp"

int main() {
  using namespace ag;

  // leaf scalar
  Variable x({2.0}, {1}, true);
  x.zero_grad();
  x.backward();                      // seed 1
  assert(x.grad()[0] == 1.0);

  // add + backward with explicit seed
  Variable a({1.0, 2.0}, {2}, true);
  Variable b({3.0, 5.0}, {2}, true);
  Variable y = add(a, b);            // [4,7]
  y.zero_grad();
  y.backward(std::vector<double>{1.0, 1.0});  // d(sum)/dy = ones
  assert(a.grad()[0] == 1.0 && a.grad()[1] == 1.0);
  assert(b.grad()[0] == 1.0 && b.grad()[1] == 1.0);

  std::cout << "[OK] basic tests passed\n";
  return 0;
}
