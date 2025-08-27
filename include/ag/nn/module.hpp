// ============================
// File: include/ag/nn/module.hpp
// ============================
#pragma once

#include <string>
#include <utility>
#include <vector>

#include "ag/core/variables.hpp"  // assumes ag::Variable lives here

namespace ag::nn {

// Minimal, framework-ready base class for neural-network modules.
// - Pure-virtual forward()
// - Parameter registration + recursive collection
// - Named children/params (for checkpoints, debugging)
// - train()/eval() with a protected on_mode_change() hook
// - zero_grad()
//
// Notes:
// * Copy/move are deleted to avoid dangling Module*/Variable* registrations.
// * Derived classes must implement _parameters() to return ONLY their own params.
// * Use register_module(name, child) to create a stable name path for state.
class Module {
public:
  Module() = default;
  virtual ~Module() = default;

  Module(const Module&)            = delete;
  Module& operator=(const Module&) = delete;
  Module(Module&&)                 = delete;
  Module& operator=(Module&&)      = delete;

  // Core forward API
  virtual Variable forward(const Variable& x) = 0;

  // Parameter utilities
  std::vector<Variable*> parameters();
  std::vector<std::pair<std::string, Variable*>> named_parameters(const std::string& prefix = "");
  void zero_grad();

  // Mode control
  void train();
  void eval();
  bool training() const;

  // Hierarchy (children)
  Module& register_module(Module& m);                       // anonymous child
  Module& register_module(const std::string& name, Module& m); // named child

  // Parameter registration (for naming/checkpointing)
  Module& register_parameter(const std::string& name, Variable& v);

  // Optional advanced access
  std::vector<Module*>& submodules() { return submodules_; }

protected:
  // Derived classes can override this to react to mode changes.
  virtual void on_mode_change() {}

  // Derived classes MUST implement this to return pointers to their own params only.
  virtual std::vector<Variable*> _parameters() = 0;

private:
  bool is_training_ = true;

  // Children for recursion
  std::vector<Module*> submodules_;
  std::vector<std::pair<std::string, Module*>> named_children_;

  // Explicitly named parameters local to *this* module
  std::vector<std::pair<std::string, Variable*>> named_params_;
};

} // namespace ag::nn