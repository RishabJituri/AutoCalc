// ============================
// File: src/ag/nn/module.cpp
// ============================
#include "ag/nn/module.hpp"
#include <algorithm>
#include <unordered_set>

namespace ag::nn {

// src/ag/nn/module.cpp
std::vector<Variable*> Module::parameters() {
  std::vector<Variable*> out;
  out.reserve(16);

  auto append = [&](const std::vector<Variable*>& src) {
    for (auto* p : src) if (p) out.push_back(p);
  };

  // Local unnamed + explicitly named
  append(_parameters());
  for (auto& [name, p] : named_params_) if (p) out.push_back(p);

  // Children (only once via submodules_)
  for (auto* child : submodules_) {
    if (!child) continue;
    auto child_params = child->parameters();
    out.insert(out.end(), child_params.begin(), child_params.end());
  }

  // Order-preserving dedup by pointer
  std::vector<Variable*> deduped;
  deduped.reserve(out.size());
  std::unordered_set<Variable*> seen;
  seen.reserve(out.size());
  for (auto* p : out) {
    if (p && seen.insert(p).second) deduped.push_back(p);
  }
  return deduped;
}




std::vector<std::pair<std::string, Variable*>> Module::named_parameters(const std::string& prefix) {
  std::vector<std::pair<std::string, Variable*>> out;

  // Local: unnamed from _parameters()
  for (auto* p : _parameters()) {
    if (!p) continue;
    out.emplace_back(prefix.empty() ? std::string("") : prefix, p);
  }

  // Local: explicitly named params
  for (auto& [name, p] : named_params_) {
    if (!p) continue;
    std::string full = prefix.empty() ? name : (prefix + (name.empty() ? std::string("") : "." + name));
    out.emplace_back(full, p);
  }

  // Named children recurse with prefix
  for (auto& [cname, child] : named_children_) {
    if (!child) continue;
    std::string child_prefix = prefix.empty() ? cname : (prefix + "." + cname);
    auto child_named = child->named_parameters(child_prefix);
    out.insert(out.end(), child_named.begin(), child_named.end());
  }

  // Anonymous children (no name prefix)
  for (auto* child : submodules_) {
    if (!child) continue;
    bool is_named = false;
    for (auto& kv : named_children_) if (kv.second == child) { is_named = true; break; }
    if (is_named) continue;

    auto child_named = child->named_parameters(prefix);
    out.insert(out.end(), child_named.begin(), child_named.end());
  }

  return out;
}

void Module::zero_grad() {
  for (auto* p : parameters()) {
    if (p) p->zero_grad();
  }
}

void Module::train() {
  is_training_ = true;
  on_mode_change();
  for (Module* sm : submodules_) if (sm) sm->train();
}

void Module::eval() {
  is_training_ = false;
  on_mode_change();
  for (Module* sm : submodules_) if (sm) sm->eval();
}

bool Module::training() const { return is_training_; }

Module& Module::register_module(Module& m) {
  submodules_.push_back(&m);
  return m;
}

Module& Module::register_module(const std::string& name, Module& m) {
  submodules_.push_back(&m);
  named_children_.push_back({name, &m});
  return m;
}

Module& Module::register_parameter(const std::string& name, Variable& v) {
  named_params_.push_back({name, &v});
  return *this;
}

} // namespace ag::nn
