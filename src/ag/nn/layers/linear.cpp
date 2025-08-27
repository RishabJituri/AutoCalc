// ============================
// File: src/ag/nn/linear.cpp
// ============================
#include "ag/nn/layers/linear.hpp"

// We assume these free functions exist in namespace ag (from your core):
//   Variable matmul(const Variable&, const Variable&);
//   Variable add(const Variable&, const Variable&);  // with row-wise broadcast for bias
namespace ag::nn {

void Linear::init_params_(double scale, unsigned long long seed) {
  const std::size_t in  = in_features_;
  const std::size_t out = out_features_;

  // W shape [In, Out]
  const std::size_t nW = in * out;
  auto wdata = randu_(nW, scale, seed);
  W_ = make_param_(wdata, {in, out});

  if (bias_) {
    auto bdata = randu_(out, scale, seed + 1);
    b_ = make_param_(bdata, {out});
  }
}

Variable Linear::forward(const Variable& x) {
  // x: [B, In], W: [In, Out] -> y: [B, Out]
  auto y = ag::matmul(x, W_);
  if (bias_) {
    // Expecting add to support row-wise broadcast of [Out] over [B,Out]
    y = ag::add(y, b_);
  }
  return y;
}

} // namespace ag::nn


// ============================
// File: include/ag/nn/sequential.hpp
// ============================
#pragma once

#include <vector>
#include <string>
#include <sstream>
#include "ag/nn/module.hpp"

namespace ag::nn {

// A simple container that applies child modules in order.
class Sequential : public Module {
public:
  Sequential() = default;

  // Register child with auto-indexed name ("0", "1", ...)
  Module& push_back(Module& m) {
    std::ostringstream os; os << layers_.size();
    layers_.push_back(&register_module(os.str(), m));
    return m;
  }

  // Register child with an explicit name
  Module& add(const std::string& name, Module& m) {
    layers_.push_back(&register_module(name, m));
    return m;
  }

  Variable forward(const Variable& x) override {
    Variable h = x;
    for (Module* m : layers_) {
      h = m->forward(h);
    }
    return h;
  }

protected:
  // No local parameters; Module::parameters() will recurse into children
  std::vector<Variable*> _parameters() override { return {}; }

private:
  std::vector<Module*> layers_;
};

} // namespace ag::nn


// ============================
// File: tests/test_nn_linear_sequential.cpp
// (Optional smoke test illustrating usage; adjust to your test framework)
// ============================
#include <cassert>
#include <vector>
#include "ag/nn/layers/linear.hpp"
#include "ag/nn/sequential.hpp"
#include "ag/core/variables.hpp"

using ag::Variable;

static Variable tensor(const std::vector<double>& data, const std::vector<std::size_t>& shape, bool req=true) {
  return Variable(data, shape, req);
}

int main() {
  using namespace ag::nn;

  // Build a tiny MLP: [3] -> [4] -> [2]
  Linear l1(3, 4);
  Linear l2(4, 2, /*bias=*/true);

  Sequential net;
  net.add("fc1", l1);
  net.add("fc2", l2);

  // Dummy batch of size 5
  std::vector<double> xdata(5 * 3, 0.1);
  Variable x = tensor(xdata, {5, 3}, /*req=*/false);

  Variable y = net.forward(x);
  auto shp = y.shape();
  assert(shp.size() == 2 && shp[0] == 5 && shp[1] == 2);

  // Params should be non-empty
  auto ps = net.parameters();
  assert(!ps.empty());

  // zero_grad should not crash
  net.zero_grad();

  return 0;
}
