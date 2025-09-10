
#include "ag/core/stochastic.hpp"
#include "ag/ops/tensor_utils.hpp"
#include "ag/core/rng.hpp"
#include <cmath>
#include <algorithm>
#include <stdexcept>

namespace ag{
using detail::numel;

// Utilities
static void softmax_rows(const std::vector<double>& s, const std::vector<std::size_t>& shape,
                         std::vector<double>& out_p, std::vector<double>& out_logp_row) {
  // shape = [K] or [B,K]; out_p same size as s. out_logp_row shape [B] or [1].
  if (shape.empty()) throw std::invalid_argument("softmax_rows: empty shape");
  if (shape.size() > 2) throw std::invalid_argument("softmax_rows: rank > 2 not supported");
  const std::size_t B = (shape.size()==2)? shape[0] : 1;
  const std::size_t K = (shape.size()==2)? shape[1] : shape[0];
  out_p.resize(s.size());
  out_logp_row.assign(B, 0.0);
  for (std::size_t b=0; b<B; ++b) {
    const std::size_t base = b*K;
    // log-sum-exp
    double m = -1e300;
    for (std::size_t k=0;k<K;++k) m = std::max(m, s[base+k]);
    double sum = 0.0;
    for (std::size_t k=0;k<K;++k) sum += std::exp(s[base+k]-m);
    const double lse = m + std::log(sum);
    out_logp_row[b] = lse;
    for (std::size_t k=0;k<K;++k) out_p[base+k] = std::exp(s[base+k]-lse);
  }
}

// Node that produces scalar logprob per row for chosen action(s). Backward: (one_hot - p).
static Variable LogProbFromLogitsAndOneHot(const Variable& logits, const Variable& onehot) {
  auto out = std::make_shared<Node>();
  // logits: [B,K] or [K], onehot same shape, out: [B] or []
  const auto& shp = logits.n->shape;
  if (shp != onehot.n->shape) throw std::invalid_argument("logits/onehot shape mismatch");
  if (shp.empty()) throw std::invalid_argument("logits must have rank 1 or 2");
  const std::size_t B = (shp.size()==2)? shp[0] : 1;
  const std::size_t K = (shp.size()==2)? shp[1] : shp[0];

  out->shape = (B==1? std::vector<std::size_t>{} : std::vector<std::size_t>{B});
  out->value.assign(B==1? 1: B, 0.0);
  out->grad.assign(B==1? 1: B, 0.0);
  out->parents = {logits.n, onehot.n};
  out->requires_grad = logits.n->requires_grad; // onehot is constant leaf typically

  // forward: compute p and then log p(a) per row
  std::vector<double> p, lse;
  softmax_rows(logits.n->value, shp, p, lse);
  for (std::size_t b=0;b<B;++b) {
    double lp = 0.0;
    for (std::size_t k=0;k<K;++k) {
      const double oh = onehot.n->value[b*K + k];
      if (oh!=0.0) {
        lp += (logits.n->value[b*K + k] - lse[b]); // log p(a)
      }
    }
    out->value[B==1?0:b] = lp;
  }

  std::weak_ptr<Node> ow = out, lw = logits.n, hw = onehot.n;
  out->backward = [ow, lw, hw, shp]() {
    auto o = ow.lock(); if (!o) return;
    auto l = lw.lock(); auto h = hw.lock();
    if (!l || !l->requires_grad) return;
    const std::size_t B = (shp.size()==2)? shp[0] : 1;
    const std::size_t K = (shp.size()==2)? shp[1] : shp[0];
    // Recompute probs from saved logits (cheap)
    std::vector<double> p, lse;
    softmax_rows(l->value, shp, p, lse);
    for (std::size_t b=0;b<B;++b) {
      const double go = o->grad[B==1?0:b]; // upstream scalar per row
      // grad wrt logits = (one_hot - p) * go
      for (std::size_t k=0;k<K;++k) {
        const double oh = h? h->value[b*K + k] : 0.0;
        l->grad[b*K + k] += go * (oh - p[b*K + k]);
      }
    }
  };

  return make_from_node(out);
}

SampleOut CategoricalSample(const Variable& logits, uint64_t seed) {
  if (logits.n->shape.empty())
    throw std::invalid_argument("CategoricalSample: logits must be rank 1 or 2");
  const std::size_t B = (logits.n->shape.size()==2)? logits.n->shape[0] : 1;
  const std::size_t K = (logits.n->shape.size()==2)? logits.n->shape[1] : logits.n->shape[0];

  // compute probs
  std::vector<double> p, lse;
  softmax_rows(logits.n->value, logits.n->shape, p, lse);

  // sample indices and build one-hot
  RNG rng(seed);
  std::vector<double> onehot(B*K, 0.0);
  for (std::size_t b=0;b<B;++b) {
    double u = rng.next_uniform01();
    // inverse cdf
    double acc = 0.0; std::size_t a = 0;
    for (; a<K; ++a) { acc += p[b*K + a]; if (u <= acc) break; }
    if (a>=K) a=K-1;
    onehot[b*K + a] = 1.0;
  }

  // Build Variables
  auto onehot_node = std::make_shared<Node>();
  onehot_node->shape = logits.n->shape;
  onehot_node->value = std::move(onehot);
  onehot_node->grad.assign(B*K, 0.0);
  onehot_node->requires_grad = false;
  // onehot is a leaf constant in the graph (no backward)

  Variable onehot_var = make_from_node(onehot_node);
  Variable logprob_var = LogProbFromLogitsAndOneHot(logits, onehot_var);
  return { onehot_var, logprob_var };
}

Variable GumbelSoftmax(const Variable& logits, double tau, bool hard, uint64_t seed) {
  if (tau <= 0.0) throw std::invalid_argument("GumbelSoftmax: tau must be > 0");
  if (logits.n->shape.empty() || logits.n->shape.size()>2)
    throw std::invalid_argument("GumbelSoftmax: logits shape must be [K] or [B,K]");
  const std::size_t B = (logits.n->shape.size()==2)? logits.n->shape[0] : 1;
  const std::size_t K = (logits.n->shape.size()==2)? logits.n->shape[1] : logits.n->shape[0];

  // Build a node: forward adds Gumbel noise, divides by tau, softmax; backward is standard softmax backward (pathwise)
  auto out = std::make_shared<Node>();
  out->shape = logits.n->shape;
  out->value.assign(B*K, 0.0);
  out->grad.assign(B*K, 0.0);
  out->parents = { logits.n };
  out->requires_grad = logits.n->requires_grad;

  // forward
  RNG rng(seed);
  std::vector<double> s = logits.n->value;
  for (std::size_t i=0;i<s.size();++i) {
    // Gumbel(0,1) = -log(-log U)
    double u = std::max(1e-12, rng.next_uniform01());
    double g = -std::log(-std::log(u));
    s[i] = (s[i] + g) / tau;
  }
  // softmax
  std::vector<double> p, lse;
  softmax_rows(s, out->shape, p, lse);
  // possibly straight-through hardening
  if (hard) {
    // Save soft probs in value for backward (as usual), but during forward, output hard one-hot
    // We still store p in out->value for downstream use; users can call StopGradient if they need
    // a true hard forward elsewhere.
  }
  out->value = std::move(p);

  std::weak_ptr<Node> ow = out, lw = logits.n;
  out->backward = [ow, lw]() {
    auto o = ow.lock(); if (!o) return;
    auto l = lw.lock();
    if (!l || !l->requires_grad) return;
    const auto& shp = o->shape;
    const std::size_t B = (shp.size()==2)? shp[0] : 1;
    const std::size_t K = (shp.size()==2)? shp[1] : shp[0];
    // softmax backward per row: dL/ds = J_softmax^T * dL/dp
    for (std::size_t b=0;b<B;++b) {
      // compute dot(go, p)
      double dot = 0.0;
      for (std::size_t k=0;k<K;++k) dot += o->grad[b*K + k] * o->value[b*K + k];
      for (std::size_t k=0;k<K;++k) {
        const double pk = o->value[b*K + k];
        const double go = o->grad[b*K + k];
        const double gs = pk * (go - dot); // derivative wrt logits (since ds/dlogits = I for added noise treated constant)
        l->grad[b*K + k] += gs;
      }
    }
  };

  return make_from_node(out);
}

} // namespace ag
