#pragma once
#include "ag/nn/module.hpp"
#include <vector>
#include <memory>
#include <utility>   // std::move

namespace ag::nn {

struct Sequential : Module {
  std::vector<std::shared_ptr<Module>> layers;

  Sequential() = default;

  // Accept std::shared_ptr<Module> args
  template <typename... Mods>
  explicit Sequential(Mods... mods) { (push(std::move(mods)), ...); }

  // Registers the submodule so train()/eval()/parameters() recurse correctly
  void push(std::shared_ptr<Module> m) {
    if (!m) return;
    register_module(*m);          // use the unnamed overload; or name it if you want
    layers.emplace_back(std::move(m));
  }

  Variable forward(const Variable& x) override {
    Variable y = x;
    for (auto& m : layers) y = m->forward(y);
    return y;
  }

protected:
  // No learnable tensors directly on Sequential itself
  std::vector<Variable*> _parameters() override { return {}; }
  void on_mode_change() override {} // nothing special here
};

} // namespace ag::nn
