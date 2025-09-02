// ============================
// File: include/ag/data/dataset.hpp
// ============================
#pragma once
#include <vector>
#include <cstddef>
#include <utility>
#include <memory>
#include <stdexcept>

#include "ag/core/variables.hpp"

namespace ag::data {

struct Example {
  ag::Variable x;
  ag::Variable y;
};

struct Dataset {
  virtual ~Dataset() = default;
  virtual std::size_t size() const = 0;
  virtual Example get(std::size_t idx) const = 0;
};

// Simple in-memory dataset of preconstructed Examples.
class InMemoryDataset final : public Dataset {
public:
  InMemoryDataset() = default;
  explicit InMemoryDataset(std::vector<Example> items) : items_(std::move(items)) {}
  std::size_t size() const override { return items_.size(); }
  Example get(std::size_t idx) const override {
    if (idx >= items_.size()) throw std::out_of_range("InMemoryDataset idx out of range");
    return items_[idx];
  }
  void push_back(Example e) { items_.push_back(std::move(e)); }
private:
  std::vector<Example> items_;
};

} // namespace ag::data
