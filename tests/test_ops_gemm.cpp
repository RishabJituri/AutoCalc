#include "test_framework.hpp"
#include "ag/ops/gemm.hpp"
#include "ag/parallel/config.hpp"
#include <vector>
#include <random>
#include <cmath>

using namespace ag::ops;
using namespace ag::parallel;

static void naive_gemm_accum(int M,int N,int K,
                             const float* A,int lda,
                             const float* B,int ldb,
                             float* C,int ldc) {
  // C += A * B  (row-major)
  for (int i=0;i<M;++i) {
    for (int j=0;j<N;++j) {
      float s = 0.f;
      for (int p=0;p<K;++p) s += A[i*lda+p] * B[p*ldb+j];
      C[i*ldc+j] += s;
    }
  }
}

static void fill(std::vector<float>& v, float lo=-1.0f, float hi=1.0f) {
  std::mt19937 rng(123);
  std::uniform_real_distribution<float> dist(lo,hi);
  for (auto& x: v) x = dist(rng);
}

static float rel_err(const std::vector<float>& a, const std::vector<float>& b) {
  float rmax=0.f;
  for (size_t i=0;i<a.size();++i) {
    float d = std::fabs(a[i]-b[i]);
    float m = std::fabs(b[i]);
    float r = d / (m + 1e-6f);
    if (r > rmax) rmax = r;
  }
  return rmax;
}

TEST("gemm/blocked_scalar_matches_naive") {
  set_max_threads(4);

  const int M=257, N=259, K=253; // force tail handling
  std::vector<float> A(M*K), B(K*N), C_ref(M*N,0.f), C(M*N,0.f);
  fill(A); fill(B);

  // Reference: C_ref += A*B
  naive_gemm_accum(M,N,K, A.data(),K, B.data(),N, C_ref.data(),N);

  // Under test: sgemm_f32 does C += A*B
  sgemm_f32(M,N,K, A.data(),K, B.data(),N, C.data(),N);

  float err = rel_err(C, C_ref);
  ASSERT_NEAR(err, 0.0, 1e-5);
}

TEST("gemm/square_sizes_various") {
  set_max_threads(4);

  for (int sz : {64, 128, 192, 256}) {
    const int M=sz, N=sz, K=sz;
    std::vector<float> A(M*K), B(K*N), C_ref(M*N,0.f), C(M*N,0.f);
    fill(A); fill(B);

    naive_gemm_accum(M,N,K, A.data(),K, B.data(),N, C_ref.data(),N);
    sgemm_f32(M,N,K, A.data(),K, B.data(),N, C.data(),N);

    float err = rel_err(C, C_ref);
    ASSERT_NEAR(err, 0.0, 1e-5);
  }
}
