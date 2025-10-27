#include "ag/nn/optim/sgd.hpp"
#include "ag/parallel/parallel_for.hpp"
#include <algorithm>
#include <vector>

namespace ag::nn {

void SGD::step(ag::nn::Module& m) {
  auto params = m.parameters();                 // std::vector<ag::Variable*>

  struct Block {
    float* w;
    float* g;
    float* v;
    std::size_t n;
  };

  std::vector<Block> blocks;
  blocks.reserve(params.size());

  // Reserve capacity in the velocity map to avoid rehashing while we take pointers
  velocity.reserve(params.size());

  // First pass: ensure velocity entries and grad buffers exist (no taking pointers yet)
  for (ag::Variable* p : params) {
    if (!p || !p->n) continue;
    ag::Node* key = p->n.get();
    const std::size_t n = p->n->value.size();
    auto &vel = velocity[key];
    if (vel.size() != n) vel.assign(n, 0.0f);
    // Ensure grad buffer exists
    if (p->n->grad.size() != n) p->n->grad.assign(n, 0.0f);
  }

  // Second pass: now take stable data() pointers into blocks
  for (ag::Variable* p : params) {
    if (!p || !p->n) continue;
    ag::Node* key = p->n.get();
    auto& vel = velocity[key];
    const std::size_t n = p->n->value.size();
    blocks.push_back(Block{ p->n->value.data(), p->n->grad.data(), vel.data(), n });
  }

  // Hoist hyperparams
  const float lr_ = lr;
  const float mu  = momentum;
  const float wd  = weight_decay;
  const bool use_nest = nesterov;

  // Serial update body for a block
  auto serial_update = [&](Block& b) {
    float* __restrict__ W = b.w;
    float* __restrict__ G = b.g;
    float* __restrict__ V = b.v;
    const std::size_t N = b.n;
    if (wd != 0.0f) {
      for (std::size_t i = 0; i < N; ++i) {
        float gi = G[i] + wd * W[i];
        float vi = (V[i] = mu * V[i] + gi);
        const float step = use_nest ? (mu * vi + gi) : vi;
        W[i] -= lr_ * step;
        G[i] = 0.0f;
      }
    } else {
      for (std::size_t i = 0; i < N; ++i) {
        float gi = G[i];
        float vi = (V[i] = mu * V[i] + gi);
        const float step = use_nest ? (mu * vi + gi) : vi;
        W[i] -= lr_ * step;
        G[i] = 0.0f;
      }
    }
  };

  const std::size_t P = blocks.size();
  const std::size_t threads = std::max<std::size_t>(1, ag::parallel::get_max_threads());
  const std::size_t inner_cutoff_elems = (1u << 16) / sizeof(float); // ~64KB

  if (P >= threads) {
    // Parallel over parameters
    ag::parallel::parallel_for(P, /*grain=*/1, [&](std::size_t b0, std::size_t b1){
      for (std::size_t bi = b0; bi < b1; ++bi) serial_update(blocks[bi]);
    });
  } else {
    // Few parameters: for large tensors, parallelize inner loop
    for (std::size_t bi = 0; bi < P; ++bi) {
      Block& b = blocks[bi];
      if (b.n < inner_cutoff_elems) { serial_update(b); continue; }
      // inner parallel update
      ag::parallel::parallel_for(b.n, /*grain=*/1024, [&](std::size_t i0, std::size_t i1){
        float* __restrict__ W = b.w;
        float* __restrict__ G = b.g;
        float* __restrict__ V = b.v;
        if (wd != 0.0f) {
          for (std::size_t i = i0; i < i1; ++i) {
            float gi = G[i] + wd * W[i];
            float vi = (V[i] = mu * V[i] + gi);
            const float step = use_nest ? (mu * vi + gi) : vi;
            W[i] -= lr_ * step;
            G[i] = 0.0f;
          }
        } else {
          for (std::size_t i = i0; i < i1; ++i) {
            float gi = G[i];
            float vi = (V[i] = mu * V[i] + gi);
            const float step = use_nest ? (mu * vi + gi) : vi;
            W[i] -= lr_ * step;
            G[i] = 0.0f;
          }
        }
      });
    }
  }
}

} // namespace ag::nn
