// ============================
// File: include/ag/data/dataloader.hpp
// ============================
#pragma once
#include <vector>
#include <cstddef>
#include <random>
#include <algorithm>

#include "ag/core/variables.hpp"
#include "ag/ops/tensor_utils.hpp"
#include "ag/data/dataset.hpp"

namespace ag::data {

struct DataLoaderOptions {
  std::size_t batch_size = 1;
  bool shuffle = false;
  bool drop_last = false;
  unsigned long long seed = 0xDEADBEEFULL;
};

struct Batch {
  ag::Variable x; // [B, ...]
  ag::Variable y; // [B, ...]
  std::size_t size = 0;
};

// Collate a group of N samples by stacking along a new leading batch dim.
// Requires all samples to share identical per-sample shape.
inline Batch collate(const std::vector<Example>& samples) {
  if (samples.empty()) return Batch{ ag::Variable({}, {0}, false), ag::Variable({}, {0}, false), 0 };
  const std::size_t N = samples.size();
  const auto& x0 = samples[0].x;
  const auto& y0 = samples[0].y;
  const auto& xshape0 = x0.shape();
  const auto& yshape0 = y0.shape();

  // Compute per-sample numels
  auto numel = [&](const std::vector<std::size_t>& sh) {
    std::size_t n = 1; for (auto d : sh) n *= d; return n;
  };
  const std::size_t x_per = numel(xshape0);
  const std::size_t y_per = numel(yshape0);

  // Validate shapes equal
  for (std::size_t i = 1; i < N; ++i) {
    if (samples[i].x.shape() != xshape0 || samples[i].y.shape() != yshape0) {
      throw std::invalid_argument("collate: sample shapes do not match");
    }
  }

  // New shapes with batch dim
  std::vector<std::size_t> bx; bx.reserve(xshape0.size()+1); bx.push_back(N); bx.insert(bx.end(), xshape0.begin(), xshape0.end());
  std::vector<std::size_t> by; by.reserve(yshape0.size()+1); by.push_back(N); by.insert(by.end(), yshape0.begin(), yshape0.end());

  std::vector<float> xv; xv.reserve(N * x_per);
  std::vector<float> yv; yv.reserve(N * y_per);
  for (const auto& s : samples) {
    const auto& vx = s.x.value(); xv.insert(xv.end(), vx.begin(), vx.end());
    const auto& vy = s.y.value(); yv.insert(yv.end(), vy.begin(), vy.end());
  }
  return Batch{ ag::Variable(xv, bx, /*requires_grad=*/false),
                ag::Variable(yv, by, /*requires_grad=*/false),
                N };
}

class DataLoader {
public:
  // Backwards-compatible constructors: accept a stack-allocated Dataset reference or raw pointer.
  // These wrap the pointer in a non-owning shared_ptr (no-op deleter) so DataLoader does not
  // attempt to delete stack objects. This preserves existing test/example usage while still
  // allowing the shared_ptr-taking API for Python bindings.
  DataLoader(Dataset& dataset, DataLoaderOptions opts = {})
    : ds_(std::shared_ptr<Dataset>(&dataset, [](Dataset*){})), opts_(opts) {
    if (opts_.batch_size < 1) throw std::invalid_argument("DataLoader: batch_size must be >= 1");
    reset();
  }

  DataLoader(Dataset* dataset, DataLoaderOptions opts = {})
    : ds_(std::shared_ptr<Dataset>(dataset, [](Dataset*){})), opts_(opts) {
    if (!dataset) throw std::invalid_argument("DataLoader: dataset pointer is null");
    if (opts_.batch_size < 1) throw std::invalid_argument("DataLoader: batch_size must be >= 1");
    reset();
  }

  // Accept a shared_ptr<Dataset> so we can hold Python-owned datasets safely
  DataLoader(std::shared_ptr<Dataset> dataset, DataLoaderOptions opts = {})
  : ds_(std::move(dataset)), opts_(opts) {
    // Sanity-check batch size
    if (opts_.batch_size < 1) throw std::invalid_argument("DataLoader: batch_size must be >= 1");
    reset();
  }

  void reset() {
    // Build indices 0..size-1 and optionally shuffle.
    const std::size_t N = ds_->size();
    indices_.resize(N);
    for (std::size_t i = 0; i < N; ++i) indices_[i] = i;
    if (opts_.shuffle) {
      // Mix epoch into seed so each reset(epoch++) produces a different permutation.
      const unsigned long long seed = opts_.seed + static_cast<unsigned long long>(epoch_);
      std::mt19937_64 rng(seed);
      std::shuffle(indices_.begin(), indices_.end(), rng);
    }
    cursor_ = 0;
  }

  // convenience rewind -> reset (increments epoch so shuffles change)
  void rewind() {
    ++epoch_;
    reset();
  }

  std::size_t size() const { return ds_->size(); }
  std::size_t remaining() const { return (cursor_ < indices_.size()) ? (indices_.size() - cursor_) : 0; }

  bool has_next() const {
    if (cursor_ >= indices_.size()) return false;
    if (opts_.drop_last) {
      const std::size_t left = indices_.size() - cursor_;
      return left >= opts_.batch_size;
    }
    return true;
  }

  Batch next() {
    const std::size_t N = ds_->size();
    if (!has_next()) {
      throw std::out_of_range("DataLoader::next(): no more batches");
    }
    std::size_t take = std::min<std::size_t>(opts_.batch_size, indices_.size() - cursor_);
    if (opts_.drop_last && take < opts_.batch_size) {
      // Should not happen due to has_next(), but guard anyway
      throw std::out_of_range("DataLoader::next(): insufficient elements for last batch (drop_last)");
    }
    // Collect indices first, then fetch samples. Advance cursor_ only after successful fetches.
    std::vector<std::size_t> take_idx; take_idx.reserve(take);
    for (std::size_t k = 0; k < take; ++k) take_idx.push_back(indices_[cursor_ + k]);

    std::vector<Example> buf; buf.reserve(take);
    for (auto idx : take_idx) {
      buf.push_back(ds_->get(idx));
    }

    // All fetches succeeded; advance cursor
    cursor_ += take;
    return collate(buf);
  }

  const std::vector<std::size_t>& indices() const { return indices_; }
  const DataLoaderOptions& options() const { return opts_; }

private:
  std::shared_ptr<Dataset> ds_;
  DataLoaderOptions opts_{};
  // epoch counter used to vary shuffle seed each rewind/reset cycle
  std::size_t epoch_ = 0;
  std::vector<std::size_t> indices_;
  std::size_t cursor_ = 0;
};

} // namespace ag::data
