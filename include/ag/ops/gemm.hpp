#pragma once
#include <cstring>
#include <stdexcept>
#include <cstddef>
#include <algorithm>
#include <vector>
#include <cmath>
#include "ag/parallel/config.hpp"
#include "ag/parallel/pool.hpp"
#include "ag/parallel/parallel_for.hpp"
#include "ag/sys/hw.hpp" // <-- added for cache_info()

// Simple blocked SGEMM using parallel_for.
// Layout: row-major A (M x K), row-major B (K x N), row-major C (M x N).
// Computes: C += A * B
#ifndef AG_GEMM_MR
#define AG_GEMM_MR 8
#endif
#ifndef AG_GEMM_NR
#define AG_GEMM_NR 8
#endif
#ifndef AG_GEMM_KC
#define AG_GEMM_KC 256
#endif
#ifndef AG_GEMM_MC
#define AG_GEMM_MC 256
#endif
#ifndef AG_GEMM_NC
#define AG_GEMM_NC 2048
#endif

namespace ag { namespace ops {

enum class Trans { N, T };
// ---- Minimal runtime tile picker (kept in this header to avoid extra files) ----
struct GemmTiles {
  int MR, NR;
  int KC, MC, NC;
};
inline GemmTiles pick_tiles_runtime(std::size_t sizeofT) {
  const auto ci   = ag::sys::cache_info();           // expects fields l1d and l2 in BYTES
  const auto L1   = std::max<std::size_t>(ci.l1d, 16*1024ul);   // fallbacks for older boxes
  const auto L2   = std::max<std::size_t>(ci.l2,  128*1024ul);
  const int  Tthr = (int)ag::parallel::get_max_threads();

  // Keep micro-kernel shape at 8x8 to match microkernel_8x8_f32
  int MR = 8, NR = 8;

  // Choose Kc so a Kc×NR slice of packed B sits ~half of L1 (leave room for A/C)
  int KC = (int)std::clamp<std::size_t>( L1 / (2 * (std::size_t)NR * sizeofT),
                                         64ul, 512ul );

  // Choose Nc so a packed B panel (Kc×Nc) fits ~60% of L2
  int NC = (int)std::max<std::size_t>( (std::size_t)NR,
                (std::size_t)((0.6 * (float)L2) / (KC * sizeofT)) );
  NC = (NC / NR) * NR; if (NC < NR) NC = NR; if (NC > 2048) NC = 2048;

  // Choose Mc so per-thread A stripe ≈ L2/(2*T). Round to MR.
  int MC = (int)std::max<std::size_t>( (std::size_t)MR,
                (std::size_t)((0.5 * (float)L2) / (std::max(1,Tthr) * KC * sizeofT)) );
  MC = (MC / MR) * MR; if (MC < MR) MC = MR; if (MC > 1024) MC = 1024;

  return {MR, NR, KC, MC, NC};
}

namespace detail {

// Pack A panel: (mc x kc) into (ceil(mc/MR)*MR x kc), padded with zeros.
inline void packA_f32(const float* A, int lda,
                      float* Ap, int mc, int kc, int mr=AG_GEMM_MR) {
  const int mb = (mc + mr - 1) / mr;
  for (int bi = 0; bi < mb; ++bi) {
    const int i0 = bi * mr;
    const int ib = std::min(mr, mc - i0);
    for (int p = 0; p < kc; ++p) {
      const float* a_col = A + (i0 * lda + p);
      for (int ii = 0; ii < ib; ii++) Ap[ii] = a_col[ii * lda];
      for (int ii = ib; ii < mr; ii++) Ap[ii] = 0.0f;
      Ap += mr;
    }
  }
}

// Safe Pack B panel: (kc x nc) into (kc x ceil(nc/NR)*NR), padded with zeros.
inline void packB_f32(const float* B, int ldb,
                      float* Bp, int kc, int nc, int nr=AG_GEMM_NR) {
  const int nb = (nc + nr - 1) / nr;   // number of NR-wide column tiles
  float* out = Bp;
  for (int bj = 0; bj < nb; ++bj) {
    const int j0 = bj * nr;            // starting col within this (pc,jc) block
    for (int p = 0; p < kc; ++p) {
      const float* brow = B + p * ldb; // row p within this (pc) block
      for (int jj = 0; jj < nr; ++jj) {
        const int j = j0 + jj;         // column within [0, nc)
        out[jj] = (j < nc) ? brow[j] : 0.0f;
      }
      out += nr;
    }
  }
}

// Scalar 8x8 micro-kernel: C(8x8) += A(8xkc) * B(kc x 8)
inline void microkernel_8x8_f32(const float* Ap, const float* Bp,
                                float* C, int ldc, int kc) {
  float acc[8][8] = {{0}};
  const float* a = Ap;
  const float* b = Bp;
  for (int p = 0; p < kc; ++p) {
    const float a0=a[0], a1=a[1], a2=a[2], a3=a[3], a4=a[4], a5=a[5], a6=a[6], a7=a[7];
    const float b0=b[0], b1=b[1], b2=b[2], b3=b[3], b4=b[4], b5=b[5], b6=b[6], b7=b[7];
    acc[0][0]+=a0*b0; acc[0][1]+=a0*b1; acc[0][2]+=a0*b2; acc[0][3]+=a0*b3; acc[0][4]+=a0*b4; acc[0][5]+=a0*b5; acc[0][6]+=a0*b6; acc[0][7]+=a0*b7;
    acc[1][0]+=a1*b0; acc[1][1]+=a1*b1; acc[1][2]+=a1*b2; acc[1][3]+=a1*b3; acc[1][4]+=a1*b4; acc[1][5]+=a1*b5; acc[1][6]+=a1*b6; acc[1][7]+=a1*b7;
    acc[2][0]+=a2*b0; acc[2][1]+=a2*b1; acc[2][2]+=a2*b2; acc[2][3]+=a2*b3; acc[2][4]+=a2*b4; acc[2][5]+=a2*b5; acc[2][6]+=a2*b6; acc[2][7]+=a2*b7;
    acc[3][0]+=a3*b0; acc[3][1]+=a3*b1; acc[3][2]+=a3*b2; acc[3][3]+=a3*b3; acc[3][4]+=a3*b4; acc[3][5]+=a3*b5; acc[3][6]+=a3*b6; acc[3][7]+=a3*b7;
    acc[4][0]+=a4*b0; acc[4][1]+=a4*b1; acc[4][2]+=a4*b2; acc[4][3]+=a4*b3; acc[4][4]+=a4*b4; acc[4][5]+=a4*b5; acc[4][6]+=a4*b6; acc[4][7]+=a4*b7;
    acc[5][0]+=a5*b0; acc[5][1]+=a5*b1; acc[5][2]+=a5*b2; acc[5][3]+=a5*b3; acc[5][4]+=a5*b4; acc[5][5]+=a5*b5; acc[5][6]+=a5*b6; acc[5][7]+=a5*b7;
    acc[6][0]+=a6*b0; acc[6][1]+=a6*b1; acc[6][2]+=a6*b2; acc[6][3]+=a6*b3; acc[6][4]+=a6*b4; acc[6][5]+=a6*b5; acc[6][6]+=a6*b6; acc[6][7]+=a6*b7;
    acc[7][0]+=a7*b0; acc[7][1]+=a7*b1; acc[7][2]+=a7*b2; acc[7][3]+=a7*b3; acc[7][4]+=a7*b4; acc[7][5]+=a7*b5; acc[7][6]+=a7*b6; acc[7][7]+=a7*b7;
    a += 8; b += 8;
  }
  for (int i = 0; i < 8; ++i) {
    float* c = C + i * ldc;
    for (int j = 0; j < 8; ++j) c[j] += acc[i][j];
  }
}

} // namespace detail

// C(MxN) += A(MxK) * B(KxN)
inline void sgemm_f32(int M,int N,int K,
                      const float* A,int lda,
                      const float* B,int ldb,
                      float* C,int ldc) {
  if (M<=0 || N<=0 || K<=0) return;

  // ---- NEW: runtime-picked tiles (keeps MR/NR at 8 to match kernel) ----
  const auto tiles = pick_tiles_runtime(sizeof(float));
  const int MR = tiles.MR, NR = tiles.NR;
  const int KC = tiles.KC, MC = tiles.MC, NC = tiles.NC;

  for (int jc = 0; jc < N; jc += NC) {
    const int nc = std::min(NC, N - jc);

    for (int pc = 0; pc < K; pc += KC) {
      const int kc = std::min(KC, K - pc);

      // Base pointer for this (pc, jc) panel in original B
      const float* Bblk = B + std::size_t(pc) * ldb + jc;

      // Pack B once per (pc,jc) into shared (read-only) buffer.
      const int nb_cols = (nc + NR - 1) / NR;
      const std::size_t Bp_elems = std::size_t(kc) * nb_cols * NR;
      std::vector<float> Bp(Bp_elems);
      detail::packB_f32(Bblk, ldb, Bp.data(), kc, nc, NR);

      for (int ic = 0; ic < M; ic += MC) {
        const int mc = std::min(MC, M - ic);
        const int mb = (mc + MR - 1) / MR;
        const int nb_tiles = (nc + NR - 1) / NR;
        const std::size_t total_tiles = std::size_t(mb) * nb_tiles;

        ag::parallel::parallel_for(total_tiles, /*grain=*/0, [&](std::size_t t0, std::size_t t1){
          // one scratch per OS thread
          struct Scratch { std::vector<float> Ap; };
          thread_local Scratch s;

          for (std::size_t t = t0; t < t1; ++t) {
            const int bi = int(t % mb);
            const int bj = int(t / mb);
            const int i0 = ic + bi * MR;
            const int j0 = jc + bj * NR;
            const int ib = std::min(MR, M - i0);
            const int jb = std::min(NR, N - j0);

            const std::size_t need = std::size_t(MR) * kc;
            if (s.Ap.size() < need) s.Ap.resize(need);

            // Pack A rows [i0, i0+ib)
            detail::packA_f32(A + std::size_t(i0) * lda + pc, lda, s.Ap.data(), ib, kc, MR);

            // B panel for this column tile (read-only)
            const float* Bp_tile = Bp.data() + std::size_t(bj) * (kc * NR);

            // Compute into temp 8x8 then add tails into C
            float Ctmp[8*8] = {0};
            detail::microkernel_8x8_f32(s.Ap.data(), Bp_tile, Ctmp, 8, kc);

            float* Cij = C + std::size_t(i0) * ldc + j0;
            for (int i = 0; i < ib; ++i) {
              float* crow = Cij + std::size_t(i) * ldc;
              const float* crow_tmp = Ctmp + i * 8;
              for (int j = 0; j < jb; ++j) crow[j] += crow_tmp[j];
            }
          }
        });
      }
    }
  }
}

// Transpose-aware sgemm_f32 overload
inline void sgemm_f32(int M, int N, int K,
                      Trans transA, Trans transB,
                      const float* A, int lda,
                      const float* B, int ldb,
                      float* C, int ldc,
                      float alpha=1.0f, float beta=1.0f)
{
  if (M <= 0 || N <= 0 || K <= 0) return;

  // Scale C by beta once up front
  if (beta != 1.0f) {
    for (int i = 0; i < M; ++i) {
      float* crow = C + std::size_t(i) * ldc;
      for (int j = 0; j < N; ++j) crow[j] *= beta;
    }
  }

  if (alpha == 0.0f) return;

  // Fast path: NN uses the tuned kernel
  if (transA == Trans::N && transB == Trans::N) {
    if (alpha == 1.0f) {
      sgemm_f32(M, N, K, A, lda, B, ldb, C, ldc);
      return;
    } else {
      // Compute T = A*B, then C += alpha*T
      std::vector<float> T(std::size_t(M) * N, 0.0f);
      sgemm_f32(M, N, K, A, lda, B, ldb, T.data(), N);
      for (int i = 0; i < M; ++i) {
        float* crow = C + std::size_t(i) * ldc;
        const float* trow = T.data() + std::size_t(i) * N;
        for (int j = 0; j < N; ++j) crow[j] += alpha * trow[j];
      }
      return;
    }
  }

  // Materialize transposed operands into small temporaries (full matrices for now)
  std::vector<float> Ause, Buse;
  const float* A_mat = A;
  const float* B_mat = B;
  int lda_use = lda, ldb_use = ldb;

  if (transA == Trans::T) {
    // Ause = transpose of A (KxM)
    Ause.resize(std::size_t(M) * K);
    for (int i = 0; i < M; ++i)
      for (int k = 0; k < K; ++k)
        Ause[std::size_t(i) * K + k] = A[std::size_t(k) * lda + i];
    A_mat = Ause.data();
    lda_use = K;
  }
  if (transB == Trans::T) {
    // Buse = transpose of B (N x K)
    Buse.resize(std::size_t(K) * N);
    for (int k = 0; k < K; ++k)
      for (int j = 0; j < N; ++j)
        Buse[std::size_t(k) * N + j] = B[std::size_t(j) * ldb + k];
    B_mat = Buse.data();
    ldb_use = N;
  }

  // Now call the NN kernel with possibly materialized A/B
  if (alpha == 1.0f) {
    sgemm_f32(M, N, K, A_mat, lda_use, B_mat, ldb_use, C, ldc);
  } else {
    std::vector<float> T(std::size_t(M) * N, 0.0f);
    sgemm_f32(M, N, K, A_mat, lda_use, B_mat, ldb_use, T.data(), N);
    for (int i = 0; i < M; ++i) {
      float* crow = C + std::size_t(i) * ldc;
      const float* trow = T.data() + std::size_t(i) * N;
      for (int j = 0; j < N; ++j) crow[j] += alpha * trow[j];
    }
  }
}

// Only a shim that forwards to the new trans overload for NN
inline void sgemm_f32(int M, int N, int K,
                      const float* A, int lda,
                      const float* B, int ldb,
                      float* C, int ldc,
                      float alpha, float beta)
{
  sgemm_f32(M, N, K, Trans::N, Trans::N, A, lda, B, ldb, C, ldc, alpha, beta);
}

}} // namespace ag::ops
