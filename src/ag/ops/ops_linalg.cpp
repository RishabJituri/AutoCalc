// Deterministic tiled matmul wired directly to ag::Variable.
// Public symbol: ag::matmul(const Variable&, const Variable&) as declared in linalg.hpp.
//
// Forward:  C = A @ B        where A:[...,M,K], B:[...,K,N] -> C:[...,M,N]
// Backward: dA = dC @ B^T    and  dB = A^T @ dC
//
// Assumptions
//  - FP32 row-major contiguous storage
//  - Determinism enforced by ScopedDeterministicParallel and fixed tile order
//
// This file intentionally avoids any intermediate "TensorView" types and integrates
// straight with ag::Variable / ag::Node (value/grad/shape/parents/backward).

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <vector>

#include "ag/core/variables.hpp"
#include "ag/ops/linalg.hpp"

#include "ag/parallel/config.hpp"
#include "ag/parallel/parallel_for.hpp"
#include "ag/parallel/per_thread.hpp"
#include "ag/parallel/pool.hpp"

#include "ag/ops/gemm.hpp"   // sgemm_f32 with (alpha,beta) overload linked via shim

namespace ag {

// ---- helpers ----
static inline std::size_t numel(const std::vector<std::size_t>& sh) {
  std::size_t n = 1;
  for (auto d : sh) n *= d;
  return n;
}

static inline int iceil_div(int x, int y) { return (x + y - 1) / y; }

// Map linear tile index to (ih, iw) with row-major order over tiles
static inline void tile_index_to_coords(std::size_t t, int th, int tw, int& ih, int& iw) {
  ih = int(t / tw);
  iw = int(t % tw);
}

// Compute base pointer offsets for row-major matrices with leading dim ld.
static inline const float* ptrA(const float* A, int lda, int i0, int k0) {
  return A + std::size_t(i0) * lda + k0;
}
static inline const float* ptrB(const float* B, int ldb, int k0, int j0) {
  return B + std::size_t(k0) * ldb + j0;
}
static inline float* ptrC(float* C, int ldc, int i0, int j0) {
  return C + std::size_t(i0) * ldc + j0;
}

struct TileParams {
  int tileM = 128;
  int tileN = 128;
  int kc    = 256;
  int grain_in_tiles = 1;               // tiles per task
  std::size_t small_cutoff = 64u*64u*64u; // MNK threshold for scalar path
};

// ---- Core tiled matmul on a single (non-batched) instance ----
// A: (M,K) lda=K;  B: (K,N) ldb=N;  C: (M,N) ldc=N
static void matmul_tiled_core(const float* A, int lda,
                              const float* B, int ldb,
                              float* C, int ldc,
                              int M, int N, int K,
                              const TileParams& tp) {
  if (M <= 0 || N <= 0 || K <= 0) return;

  const std::size_t mnk = std::size_t(M) * std::size_t(N) * std::size_t(K);
  if (mnk < tp.small_cutoff) {
    // scalar deterministic path: C = A*B
    for (int i = 0; i < M; ++i) {
      float* crow = C + std::size_t(i) * ldc;
      for (int j = 0; j < N; ++j) crow[j] = 0.0f;
    }
    for (int k = 0; k < K; ++k) {
      const float* Ak = A + std::size_t(k);
      const float* Bk = B + std::size_t(k) * ldb;
      for (int i = 0; i < M; ++i) {
        const float aik = Ak[std::size_t(i) * lda];
        float* crow = C + std::size_t(i) * ldc;
        for (int j = 0; j < N; ++j) crow[j] += aik * Bk[j];
      }
    }
    return;
  }

  const int tileM = tp.tileM;
  const int tileN = tp.tileN;
  const int kc    = tp.kc;

  const int th = iceil_div(M, tileM);
  const int tw = iceil_div(N, tileN);

  // Outer blocking over column tiles (j / tileN) and reduction over K so B panels
  // (pc,jc) can be packed once and reused across row-tiles.
  for (int j0_tile = 0; j0_tile < tw; ++j0_tile) {
    const int j0 = j0_tile * tileN;
    const int j1 = std::min(j0 + tileN, N);
    const int Nb = j1 - j0;
    if (Nb <= 0) continue;

    for (int p0 = 0; p0 < K; p0 += kc) {
      const int p1 = std::min(p0 + kc, K);
      const int Kb = p1 - p0;
      if (Kb <= 0) continue;

      // Pack B once for this (p0, j0) panel into Bp
      const float* Bblk = ptrB(B, ldb, p0, j0); // (Kb x Nb)

      // Get microkernel NR from gemm macros
#ifdef AG_GEMM_NR
      const int NR = AG_GEMM_NR;
#else
      const int NR = 8;
#endif
      const int nb_cols = (Nb + NR - 1) / NR;
      const std::size_t Bp_elems = std::size_t(Kb) * nb_cols * NR;
      std::vector<float> Bp(Bp_elems);
      ag::ops::detail::packB_f32(Bblk, ldb, Bp.data(), Kb, Nb, NR);

      // Parallelize over row tile index (ih)
      const std::size_t total_row_tiles = std::size_t(th);
      ag::parallel::parallel_for(total_row_tiles, tp.grain_in_tiles, [&](std::size_t t0, std::size_t t1){
        for (std::size_t t = t0; t < t1; ++t) {
          const int ih = int(t);
          const int i0 = ih * tileM;
          const int i1 = std::min(i0 + tileM, M);
          const int Mb = i1 - i0;
          if (Mb <= 0) continue;

          const int MR = (int)AG_GEMM_MR;
          const int mb = (Mb + MR - 1) / MR;
          const int nb_tiles = (Nb + NR - 1) / NR;

          for (int bi = 0; bi < mb; ++bi) {
            const int sub_i0 = i0 + bi * MR;
            const int ib = std::min(MR, M - sub_i0);

            for (int bj = 0; bj < nb_tiles; ++bj) {
              const int sub_j0 = j0 + bj * NR;
              const int jb = std::min(NR, N - sub_j0);

              // Pointers into A and C
              const float* Ablk = ptrA(A, lda, sub_i0, p0); // (ib x Kb)
              float* Cblk = ptrC(C, ldc, sub_i0, sub_j0);   // (ib x jb)

              // Bp_tile points at (Kb x NR) block for this column tile
              const float* Bp_tile = Bp.data() + std::size_t(bj) * (std::size_t(Kb) * NR);

              // Single-threaded leaf call: packs A (per-thread) and runs microkernel
              ag::ops::detail::sgemm_packedB_f32(ib, jb, Kb, Ablk, lda, Bp_tile, Cblk, ldc,
                                                 1.0f, (p0==0 ? 0.0f : 1.0f));
            }
          }
        }
      });
    }
  }
}

// ---- Batched matmul with broadcasting over leading dims ----
static void matmul_batched_core(const float* A, const std::vector<std::size_t>& Ashape,
                                const float* B, const std::vector<std::size_t>& Bshape,
                                float* C, const std::vector<std::size_t>& Cshape,
                                const TileParams& tp) {
  assert(Ashape.size() >= 2 && Bshape.size() >= 2 && Cshape.size() >= 2);
  const int M = int(Cshape[Cshape.size()-2]);
  const int N = int(Cshape[Cshape.size()-1]);
  const int K = int(Ashape[Ashape.size()-1]);
  assert(int(Bshape[Bshape.size()-2]) == K);

  const size_t Ar = Ashape.size();
  const size_t Br = Bshape.size();
  const size_t Cr = Cshape.size();
  const size_t maxr = std::max(Ar, Br);

  std::vector<std::size_t> Ab(maxr, 1), Bb(maxr, 1), Cb(maxr, 1);
  for (size_t i = 0; i < maxr; ++i) {
    if (i < Ar) Ab[i] = Ashape[Ar - maxr + i];
    if (i < Br) Bb[i] = Bshape[Br - maxr + i];
    if (i < Cr) Cb[i] = Cshape[Cr - maxr + i];
  }

  const size_t batch_dims = maxr - 2;
  size_t Bcount = 1;
  for (size_t i = 0; i < batch_dims; ++i) {
    const size_t ad = Ab[i], bd = Bb[i], cd = Cb[i];
    if (!((ad == cd || ad == 1) && (bd == cd || bd == 1)))
      throw std::invalid_argument("matmul: incompatible broadcast batch dims");
    Bcount *= Cb[i];
  }

  const int lda = K;
  const int ldb = N;
  const int ldc = N;

  auto strides_for = [](const std::vector<std::size_t>& dims) {
    std::vector<std::size_t> st(dims.size(), 0);
    if (dims.empty()) return st;
    st.back() = 1;
    for (int i = int(dims.size()) - 2; i >= 0; --i) st[i] = st[i+1] * dims[i+1];
    return st;
  };
  std::vector<std::size_t> Abatch = Ab, Bbatch = Bb, Cbatch = Cb;
  Abatch.resize(batch_dims);
  Bbatch.resize(batch_dims);
  Cbatch.resize(batch_dims);
  auto Astr = strides_for(Abatch);
  auto Bstr = strides_for(Bbatch);
  auto Cstr = strides_for(Cbatch);

  auto base_incr = [](const std::vector<std::size_t>& dims, const std::vector<std::size_t>& strides) {
    std::vector<std::size_t> inc(strides.size(), 0);
    for (size_t i = 0; i < strides.size(); ++i) inc[i] = (dims[i] == 1) ? 0 : strides[i];
    return inc;
  };
  auto Aincr = base_incr(Abatch, Astr);
  auto Bincr = base_incr(Bbatch, Bstr);
  auto Cincr = base_incr(Cbatch, Cstr);

  for (size_t b = 0; b < Bcount; ++b) {
    size_t rem = b;
    std::size_t Abase = 0, Bbase = 0, Cbase = 0;
    for (size_t i = 0; i < batch_dims; ++i) {
      const size_t idx = rem / Cstr[i];
      rem = rem % Cstr[i];
      Abase += Aincr[i] * idx;
      Bbase += Bincr[i] * idx;
      Cbase += Cincr[i] * idx;
    }

    const float* Abptr = A + Abase * (std::size_t)M * (std::size_t)K;
    const float* Bbptr = B + Bbase * (std::size_t)K * (std::size_t)N;
    float*       Cbptr = C + Cbase * (std::size_t)M * (std::size_t)N;

    ag::parallel::ScopedDeterministicParallel scope_guard;
    matmul_tiled_core(Abptr, lda, Bbptr, ldb, Cbptr, ldc, M, N, K, tp);
  }
}

// ---- Public API: matmul on Variables (forward + autograd wiring) ----
Variable matmul(const Variable& A, const Variable& B) {
  if (A.shape().size() < 2 || B.shape().size() < 2)
    throw std::invalid_argument("matmul: rank < 2");

  const auto& Ash = A.shape();
  const auto& Bsh = B.shape();
  const std::size_t M = Ash[Ash.size()-2];
  const std::size_t K = Ash[Ash.size()-1];
  const std::size_t Kb = Bsh[Bsh.size()-2];
  const std::size_t N = Bsh[Bsh.size()-1];
  if (K != Kb) throw std::invalid_argument("matmul: inner dims mismatch");

  // Broadcast batch dims
  auto broadcast_shape = [](std::vector<std::size_t> A, std::vector<std::size_t> B) {
    const std::size_t r = std::max(A.size(), B.size());
    A.insert(A.begin(), r - A.size(), 1);
    B.insert(B.begin(), r - B.size(), 1);
    std::vector<std::size_t> out(r, 1);
    for (std::size_t i = 0; i < r; ++i) {
      if (A[i] == B[i] || A[i] == 1) out[i] = B[i];
      else if (B[i] == 1) out[i] = A[i];
      else throw std::invalid_argument("matmul: incompatible broadcast dims");
    }
    return out;
  };
  std::vector<std::size_t> Ab(Ash.begin(), Ash.end()-2);
  std::vector<std::size_t> Bb(Bsh.begin(), Bsh.end()-2);
  auto Cshape = broadcast_shape(Ab, Bb);
  Cshape.push_back(M);
  Cshape.push_back(N);

  // Allocate output
  std::vector<float> out(numel(Cshape), 0.0f);

  // Run deterministic forward
  TileParams tp{};
  {
    ag::parallel::ScopedDeterministicParallel scope_guard;
    matmul_batched_core(A.value().data(), Ash, B.value().data(), Bsh, out.data(), Cshape, tp);
  }

  const bool req = (A.requires_grad() || B.requires_grad()) && ag::is_grad_enabled();
  Variable C(out, Cshape, req);

  if (req) {
    C.n->parents = { A.n, B.n };
    C.n->backward = [An=A.n, Bn=B.n, Cn=C.n, Ash, Bsh, Cshape, tp]() {
      // Ensure grads allocated
      if (An->grad.size() != An->value.size()) An->grad.assign(An->value.size(), 0.0f);
      if (Bn->grad.size() != Bn->value.size()) Bn->grad.assign(Bn->value.size(), 0.0f);
      // NOTE: do not zero parent grads here — autograd should accumulate into
      // existing parent->grad across multiple downstream contributions. The GEMM
      // calls below always use beta=1.0 to accumulate into parents.

      // dC view
      const float* dC = Cn->grad.data();

      // ---- dA = dC @ B^T ----
      {
        // Shapes: A(...,M,K), B(...,K,N), dC(...,M,N) -> dA(...,M,K)
        const int M = int(Ash[Ash.size()-2]);
        const int K = int(Ash[Ash.size()-1]);
        const int N = int(Bsh[Bsh.size()-1]);

        // Broadcast batch dims as in forward
        auto strides_for = [](const std::vector<std::size_t>& dims) {
          std::vector<std::size_t> st(dims.size(), 0);
          if (dims.empty()) return st;
          st.back() = 1;
          for (int i = int(dims.size()) - 2; i >= 0; --i) st[i] = st[i+1] * dims[i+1];
          return st;
        };
        auto broadcast_shape = [](std::vector<std::size_t> A, std::vector<std::size_t> B) {
          const std::size_t r = std::max(A.size(), B.size());
          A.insert(A.begin(), r - A.size(), 1);
          B.insert(B.begin(), r - B.size(), 1);
          std::vector<std::size_t> out(r, 1);
          for (std::size_t i = 0; i < r; ++i) {
            if (A[i] == B[i] || A[i] == 1) out[i] = B[i];
            else if (B[i] == 1) out[i] = A[i];
            else throw std::invalid_argument("matmul: incompatible broadcast dims");
          }
          return out;
        };

        std::vector<std::size_t> Ab(Ash.begin(), Ash.end()-2);
        std::vector<std::size_t> Bb(Bsh.begin(), Bsh.end()-2);
        auto Batch = broadcast_shape(Ab, Bb);
        const size_t batch_dims = Batch.size();
        size_t Bcount = 1; for (auto d: Batch) Bcount *= d;

        // Batch strides over (...)
        auto Bstr = strides_for(Batch);

        // Leading dims
        const int lda_dC = N;     // dC (M,N)
        const int ldb_B = N;      // B (K,N)
        const int ldc_dA = K;     // dA (M,K)

        // Iterate batches lexicographically
        for (size_t b = 0; b < Bcount; ++b) {
          // decode b into multi-index; compute per-input base offsets honoring broadcast
          size_t rem = b;
          std::vector<std::size_t> idx(batch_dims, 0);
          for (size_t i = 0; i < batch_dims; ++i) { idx[i] = rem / Bstr[i]; rem %= Bstr[i]; }

          auto offset_of = [&](const std::vector<std::size_t>& full, const std::vector<std::size_t>& batch) {
            // Map idx into the input's leading dims (with broadcasting dims==1 collapsing offset to 0)
            const size_t in_r = full.size() - 2; // number of batch dims in this tensor
            std::size_t off = 0, mul = 1;
            for (int i = int(batch_dims)-1; i >= 0; --i) {
              const size_t dim = (i < (int)in_r) ? full[i] : 1;
              const size_t id = (dim == 1) ? 0 : idx[i];
              off += id * mul;
              mul *= dim;
            }
            return off;
          };

          const std::size_t Aoff = offset_of(Ash, Batch) * (std::size_t)M * (std::size_t)K;
          const std::size_t Boff = offset_of(Bsh, Batch) * (std::size_t)K * (std::size_t)N;
          const std::size_t Coff = b * (std::size_t)M * (std::size_t)N;

          const float* dCptr = dC + Coff;
          const float* Bptr  = Bn->value.data() + Boff;
          float*       dAptr = An->grad.data() + Aoff;

          TileParams tp{};
          ag::parallel::ScopedDeterministicParallel scope_guard;

          // Tile over (M,K), reduce over N in kc panels
          const int tileM = tp.tileM, tileK = tp.tileN, kc = tp.kc;
          const int th = iceil_div(M, tileM);
          const int tw = iceil_div(K, tileK);
          const std::size_t T = std::size_t(th) * tw;

          ag::parallel::parallel_for(T, tp.grain_in_tiles, [&](std::size_t t0, std::size_t t1) {
            for (std::size_t t = t0; t < t1; ++t) {
              int ih, iw; tile_index_to_coords(t, th, tw, ih, iw);
              const int i0 = ih * tileM;
              const int k0 = iw * tileK;
              const int i1 = std::min(i0 + tileM, M);
              const int k1 = std::min(k0 + tileK, K);
              const int Mb = i1 - i0;
              const int Kb = k1 - k0;
              if (Mb <= 0 || Kb <= 0) continue;

              for (int j0 = 0; j0 < N; j0 += kc) {
                const int j1 = std::min(j0 + kc, N);
                const int Nb = j1 - j0;

                const float* Ablk = dCptr + std::size_t(i0) * lda_dC + j0; // (Mb x Nb)
                const float* Bblk = Bptr  + std::size_t(k0) * ldb_B + j0;  // (Kb x Nb)
                float*       Cblk = dAptr + std::size_t(i0) * ldc_dA + k0; // (Mb x Kb)

                const float alpha = 1.0f;
                const float beta  = 1.0f; // always accumulate into parent grad
                // Use GEMM with (TransA=N, TransB=T)
                ag::ops::sgemm_f32(Mb, Kb, Nb, ag::ops::Trans::N, ag::ops::Trans::T,
                                   Ablk, lda_dC, Bblk, ldb_B, Cblk, ldc_dA, alpha, beta);
              }
            }
          });
        }
      }

      // ---- dB = A^T @ dC ----
      {
        const int M = int(Ash[Ash.size()-2]);
        const int K = int(Ash[Ash.size()-1]);
        const int N = int(Bsh[Bsh.size()-1]);

        // Recompute batch broadcast as above
        auto strides_for = [](const std::vector<std::size_t>& dims) {
          std::vector<std::size_t> st(dims.size(), 0);
          if (dims.empty()) return st;
          st.back() = 1;
          for (int i = int(dims.size()) - 2; i >= 0; --i) st[i] = st[i+1] * dims[i+1];
          return st;
        };
        auto broadcast_shape = [](std::vector<std::size_t> A, std::vector<std::size_t> B) {
          const std::size_t r = std::max(A.size(), B.size());
          A.insert(A.begin(), r - A.size(), 1);
          B.insert(B.begin(), r - B.size(), 1);
          std::vector<std::size_t> out(r, 1);
          for (std::size_t i = 0; i < r; ++i) {
            if (A[i] == B[i] || A[i] == 1) out[i] = B[i];
            else if (B[i] == 1) out[i] = A[i];
            else throw std::invalid_argument("matmul: incompatible broadcast dims");
          }
          return out;
        };

        std::vector<std::size_t> Ab(Ash.begin(), Ash.end()-2);
        std::vector<std::size_t> Bb(Bsh.begin(), Bsh.end()-2);
        auto Batch = broadcast_shape(Ab, Bb);
        const size_t batch_dims = Batch.size();
        size_t Bcount = 1; for (auto d: Batch) Bcount *= d;
        auto Bstr = strides_for(Batch);

        const int lda_A = K;    // A (M,K)
        const int ldb_dC = N;   // dC  (M,N)
        const int ldc_dB = N;   // dB  (K,N)

        for (size_t b = 0; b < Bcount; ++b) {
          size_t rem = b;
          std::vector<std::size_t> idx(batch_dims, 0);
          for (size_t i = 0; i < batch_dims; ++i) { idx[i] = rem / Bstr[i]; rem %= Bstr[i]; }

          auto offset_of = [&](const std::vector<std::size_t>& full, const std::vector<std::size_t>& batch) {
            const size_t in_r = full.size() - 2;
            std::size_t off = 0, mul = 1;
            for (int i = int(batch_dims)-1; i >= 0; --i) {
              const size_t dim = (i < (int)in_r) ? full[i] : 1;
              const size_t id = (dim == 1) ? 0 : idx[i];
              off += id * mul;
              mul *= dim;
            }
            return off;
          };

          const std::size_t Aoff = offset_of(Ash, Batch) * (std::size_t)M * (std::size_t)K;
          const std::size_t Boff = offset_of(Bsh, Batch) * (std::size_t)K * (std::size_t)N;
          const std::size_t Coff = b * (std::size_t)M * (std::size_t)N;

          const float* Aptr  = An->value.data() + Aoff; // (M,K) original A
          const float* dCptr = dC + Coff;
          float*       dBptr = Bn->grad.data() + Boff;

          TileParams tp{};
          ag::parallel::ScopedDeterministicParallel scope_guard;

          // Tile over (K,N), reduce over M in kc panels
          const int tileK = tp.tileM, tileN = tp.tileN, kc = tp.kc;
          const int th = iceil_div(K, tileK);
          const int tw = iceil_div(N, tileN);
          const std::size_t T = std::size_t(th) * tw;

          ag::parallel::parallel_for(T, tp.grain_in_tiles, [&](std::size_t t0, std::size_t t1) {
            for (std::size_t t = t0; t < t1; ++t) {
              int kh, jw; tile_index_to_coords(t, th, tw, kh, jw);
              const int k0 = kh * tileK;
              const int j0 = jw * tileN;
              const int k1 = std::min(k0 + tileK, K);
              const int j1 = std::min(j0 + tileN, N);
              const int Kb = k1 - k0;
              const int Nb = j1 - j0;
              if (Kb <= 0 || Nb <= 0) continue;

              for (int i0 = 0; i0 < M; i0 += kc) {
                const int i1 = std::min(i0 + kc, M);
                const int Mb = i1 - i0;

                const float* Ablk = Aptr  + std::size_t(i0) * lda_A + k0;     // (Mb x Kb) row-major for A
                const float* Bblk = dCptr + std::size_t(i0) * ldb_dC + j0;   // (Mb x Nb) row-major
                float*       Cblk = dBptr + std::size_t(k0) * ldc_dB + j0;   // (Kb x Nb) row-major

                const float alpha = 1.0f;
                const float beta  = 1.0f; // always accumulate into parent grad
                // Use GEMM with (TransA=T, TransB=N)
                ag::ops::sgemm_f32(Kb, Nb, Mb, ag::ops::Trans::T, ag::ops::Trans::N,
                                   Ablk, lda_A, Bblk, ldb_dC, Cblk, ldc_dB, alpha, beta);
              }
            }
          });
        }
      }
    };
  }

  return C;
}

// Materialized transpose of the last two dimensions. Forward copies data with
// last-two dims swapped. Backward accumulates the transposed gradient into the
// parent variable's grad (does not overwrite existing grad, respects accumulation).
Variable transpose(const Variable& A) {
  if (A.shape().size() < 2) throw std::invalid_argument("transpose: rank < 2");
  const auto Ash = A.shape();
  const size_t r = Ash.size();
  const size_t M = Ash[r-2];
  const size_t K = Ash[r-1];

  // Build output shape by swapping last two dims
  std::vector<std::size_t> OutSh = Ash;
  OutSh[r-2] = K;
  OutSh[r-1] = M;

  const size_t batch_dims = (r >= 2) ? (r - 2) : 0;
  size_t batch_count = 1;
  for (size_t i = 0; i < batch_dims; ++i) batch_count *= Ash[i];

  std::vector<float> out(batch_count * M * K, 0.0f);
  const float* Aptr = A.value().data();
  float* Outptr = out.data();

  // For each batch, transpose the (M x K) matrix into (K x M)
  for (size_t b = 0; b < batch_count; ++b) {
    const size_t Abase = b * (M * K);
    const size_t Obase = b * (M * K); // same total elems per batch
    for (size_t i = 0; i < M; ++i) {
      const size_t Ai = Abase + i * K;
      for (size_t j = 0; j < K; ++j) {
        // A[b, i, j] -> Out[b, j, i]
        Outptr[Obase + j * M + i] = Aptr[Ai + j];
      }
    }
  }

  const bool req = A.requires_grad() && ag::is_grad_enabled();
  Variable C(out, OutSh, req);
  if (req) {
    C.n->parents = { A.n };
    C.n->backward = [An = A.n, Cn = C.n, Ash, OutSh]() {
      // Ensure parent grad allocated
      if (An->grad.size() != An->value.size()) An->grad.assign(An->value.size(), 0.0f);
      const float* dC = Cn->grad.data();

      const size_t r = Ash.size();
      const size_t M = Ash[r-2];
      const size_t K = Ash[r-1];
      const size_t batch_dims = (r >= 2) ? (r - 2) : 0;
      size_t batch_count = 1;
      for (size_t i = 0; i < batch_dims; ++i) batch_count *= Ash[i];

      for (size_t b = 0; b < batch_count; ++b) {
        const size_t Abase = b * (M * K);
        const size_t Obase = b * (M * K);
        for (size_t i = 0; i < M; ++i) {
          for (size_t j = 0; j < K; ++j) {
            // dA[b,i,j] += dC[b,j,i]
            An->grad[Abase + i * K + j] += dC[Obase + j * M + i];
          }
        }
      }
    };
  }
  return C;
}

} // namespace ag
