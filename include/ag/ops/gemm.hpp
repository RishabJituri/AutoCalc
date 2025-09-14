#pragma once
#include <cstddef>
#include <algorithm>
#include <vector>
#include <cmath>
#include "ag/parallel/config.hpp"
#include "ag/parallel/pool.hpp"
#include "ag/parallel/parallel_for.hpp"

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
      for (int ii = 0; ii < ib; ++ii) Ap[ii] = a_col[ii * lda];
      for (int ii = ib; ii < mr; ++ii) Ap[ii] = 0.0f;
      Ap += mr;
    }
  }
}

// Safe Pack B panel: (kc x nc) into (kc x ceil(nc/NR)*NR), padded with zeros.
// Reads are guarded so we never step past the row end on the last tile.
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

  const int MR = AG_GEMM_MR, NR = AG_GEMM_NR;
  const int KC = AG_GEMM_KC, MC = AG_GEMM_MC, NC = AG_GEMM_NC;

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

}} // namespace ag::ops
