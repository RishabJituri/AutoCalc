#pragma once
#include "ag/data/dataset.hpp"
#include <functional>

namespace ag::data {

using Transform = std::function<Example(const Example&)>;

struct Compose {
  std::vector<Transform> ts;
  Example operator()(const Example& e) const {
    Example out = e;
    for (auto& t : ts) out = t(out);
    return out;
  }
};

inline Transform ToFloatScale(double scale) {
  return [scale](const Example& e){
    Example o = e;
    for (auto& v : o.x.n->value) v = v * scale;
    return o;
  };
}

inline Transform Normalize(double mean, double std) {
  return [mean,std](const Example& e){
    Example o = e;
    for (auto& v : o.x.n->value) v = (v - mean) / std;
    return o;
  };
}

struct TransformDataset : Dataset {
  std::shared_ptr<Dataset> base;
  Transform tfm;
  explicit TransformDataset(std::shared_ptr<Dataset> base, Transform tfm)
    : base(std::move(base)), tfm(std::move(tfm)) {}

  std::size_t size() const override { return base->size(); }
  Example get(std::size_t idx) const override { return tfm(base->get(idx)); }
};

} // namespace ag::data
