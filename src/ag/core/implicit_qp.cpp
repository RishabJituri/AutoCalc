#include "ag/core/implicit_qp.hpp"
#include <stdexcept>
#include <cmath>
#include <algorithm>

namespace ag {

// Tiny dense linear algebra helpers (row-major)
static void cholesky_spd(std::vector<float>& M, std::size_t n) {
  // In-place Cholesky: M = L*L^T, lower stored in M. Throws if not PD.
  for (std::size_t i=0;i<n;++i) {
    for (std::size_t j=0;j<=i;++j) {
      float s = M[i*n + j];
      for (std::size_t k=0;k<j;++k) s -= M[i*n + k]*M[j*n + k];
      if (i==j) {
        if (s <= 0.0f) throw std::runtime_error("Cholesky failed (matrix not PD)");
        M[i*n + j] = std::sqrt(s);
      } else {
        M[i*n + j] = s / M[j*n + j];
      }
    }
    // zero upper
    for (std::size_t j=i+1;j<n;++j) M[i*n + j] = 0.0f;
  }
}
static void chol_solve_lower(const std::vector<float>& L, std::size_t n,
                             const std::vector<float>& b, std::vector<float>& y) {
  y.assign(n, 0.0f);
  for (std::size_t i=0;i<n;++i) {
    float s = b[i];
    for (std::size_t k=0;k<i;++k) s -= L[i*n + k]*y[k];
    y[i] = s / L[i*n + i];
  }
}
static void chol_solve_upper(const std::vector<float>& L, std::size_t n,
                             const std::vector<float>& y, std::vector<float>& x) {
  x.assign(n, 0.0f);
  for (int ii=int(n)-1; ii>=0; --ii) {
    std::size_t i = std::size_t(ii);
    float s = y[i];
    for (std::size_t k=i+1;k<n;++k) s -= L[k*n + i]*x[k];
    x[i] = s / L[i*n + i];
  }
}
static void chol_solve(const std::vector<float>& L, std::size_t n,
                       const std::vector<float>& b, std::vector<float>& x_out) {
  std::vector<float> y;
  chol_solve_lower(L, n, b, y);
  chol_solve_upper(L, n, y, x_out);
}

// Compute H^{-1} * v via Cholesky of H (assuming SPD). Reuse factor by passing L factored in H_L if non-empty.
static void solve_Hinv_times(const std::vector<float>& H_L, std::size_t n,
                             const std::vector<float>& v, std::vector<float>& x) {
  chol_solve(H_L, n, v, x);
}

// Build S = A H^{-1} A^T given H factor L (lower)
static void build_S(const std::vector<float>& H_L, std::size_t n,
                    const std::vector<float>& A, std::size_t m,
                    std::vector<float>& S) {
  S.assign(m*m, 0.0f);
  // S_ij = e_i^T A H^{-1} A^T e_j = (row i of A) * H^{-1} * (row j of A)^T
  // Compute columns of H^{-1} A^T first
  std::vector<float> tmp(n), col(n);
  for (std::size_t j=0;j<m;++j) {
    // v = A^T e_j  -> take row j of A and treat as v on columns
    for (std::size_t i=0;i<n;++i) tmp[i] = A[j*n + i];
    // w = H^{-1} v
    solve_Hinv_times(H_L, n, tmp, col);
    // Fill S[:,j] = A * w
    for (std::size_t i=0;i<m;++i) {
      float s = 0.0f;
      for (std::size_t k=0;k<n;++k) s += A[i*n + k] * col[k];
      S[i*m + j] = s;
    }
  }
}

// Solve equality-constrained QP via Schur complement
static void qp_solve_eq_forward(const std::vector<float>& H_in, std::size_t n,
                                const std::vector<float>& q, 
                                const std::vector<float>& A, std::size_t m,
                                const std::vector<float>& b,
                                std::vector<float>& y_star,
                                std::vector<float>& lambda_star,
                                std::vector<float>& H_L_cache) {
  // Make H PD (tiny ridge)
  std::vector<float> H = H_in;
  for (std::size_t i=0;i<n;++i) H[i*n + i] += 1e-8f;
  // Cholesky
  cholesky_spd(H, n); // now H is L
  H_L_cache = H;

  // Solve for λ: (A H^{-1} A^T) λ = -(b + A H^{-1} q)
  std::vector<float> S;
  build_S(H_L_cache, n, A, m, S);

  // rhs = -(b + A H^{-1} q)
  std::vector<float> Hinv_q, rhs(m, 0.0f);
  solve_Hinv_times(H_L_cache, n, q, Hinv_q);
  for (std::size_t i=0;i<m;++i) {
    float s = -b[i];
    for (std::size_t k=0;k<n;++k) s -= A[i*n + k] * Hinv_q[k];
    rhs[i] = s;
  }
  // Solve S λ = rhs (S SPD)
  std::vector<float> S_L = S;
  cholesky_spd(S_L, m);
  chol_solve(S_L, m, rhs, lambda_star);

  // y = - H^{-1}(q + A^T λ)
  std::vector<float> tmp(n, 0.0f);
  for (std::size_t i=0;i<n;++i) {
    float s = q[i];
    for (std::size_t j=0;j<m;++j) s += A[j*n + i] * lambda_star[j];
    tmp[i] = s;
  }
  solve_Hinv_times(H_L_cache, n, tmp, y_star);
  for (auto& v : y_star) v = -v;
}

Variable QPSolveEq(const Variable& H, const Variable& q,
                   const Variable& A, const Variable& b,
                   float eps_pd) {
  (void)eps_pd; // parameter currently unused; forward adds a tiny ridge internally

  // Shapes
  if (H.n->shape.size()!=2 || H.n->shape[0]!=H.n->shape[1])
    throw std::invalid_argument("QPSolveEq: H must be square [N,N]");
  const std::size_t N = H.n->shape[0];
  if (q.n->shape.size()!=1 || q.n->shape[0]!=N)
    throw std::invalid_argument("QPSolveEq: q shape must be [N] to match H");
  if (A.n->shape.size()!=2 || A.n->shape[1]!=N)
    throw std::invalid_argument("QPSolveEq: A shape must be [M,N]");
  const std::size_t M = A.n->shape[0];
  if (b.n->shape.size()!=1 || b.n->shape[0]!=M)
    throw std::invalid_argument("QPSolveEq: b shape must be [M] to match A");

  auto out = std::make_shared<Node>();
  out->shape = {N};
  out->value.assign(N, 0.0f);
  out->grad.assign(N, 0.0f);
  out->parents = {H.n, q.n, A.n, b.n};
  out->requires_grad = (H.n->requires_grad || q.n->requires_grad || A.n->requires_grad || b.n->requires_grad);

  // forward
  std::vector<float> y_star, lambda_star, H_L;
  qp_solve_eq_forward(H.n->value, N, q.n->value, A.n->value, M, b.n->value, y_star, lambda_star, H_L);
  out->value = y_star;

  // Cache needed data for backward (store copies in lambda-captured shared_ptrs)
  auto Hval = H.n->value; // keep original H (before ridge)
  auto Aval = A.n->value;
  auto qval = q.n->value;
  auto bval = b.n->value;
  std::weak_ptr<Node> ow = out, Hw = H.n, qw = q.n, Aw = A.n, bw = b.n;
  out->backward = [ow, Hw, qw, Aw, bw, Hval, Aval, qval, bval, N, M]() {
    auto o = ow.lock(); if (!o) return;
    auto Hn = Hw.lock(); auto qn = qw.lock(); auto An = Aw.lock(); auto bn = bw.lock();
    if ((!Hn || !Hn->requires_grad) && (!qn || !qn->requires_grad) &&
        (!An || !An->requires_grad) && (!bn || !bn->requires_grad)) return;

    // Recompute forward solve (cheap, small dims assumed). Let forward add ridge once.
    std::vector<float> y_star, lambda_star, H_L;
    qp_solve_eq_forward(Hval, N, qval, Aval, M, bval, y_star, lambda_star, H_L);

    // Solve adjoint (KKT)^T * [u_y; u_lambda] = [o->grad; 0]
    // Schur: (A H^{-1} A^T) u_lambda = A H^{-1} bar_y
    std::vector<float> S;
    build_S(H_L, N, Aval, M, S);
    // rhs_lambda = A H^{-1} bar_y
    std::vector<float> Hinv_bar_y, rhs_lambda(M, 0.0f);
    solve_Hinv_times(H_L, N, o->grad, Hinv_bar_y);
    for (std::size_t i=0;i<M;++i) {
      float s = 0.0f;
      for (std::size_t k=0;k<N;++k) s += Aval[i*N + k] * Hinv_bar_y[k];
      rhs_lambda[i] = s;
    }
    // Solve S u_lambda = rhs
    std::vector<float> S_L = S, u_lambda;
    cholesky_spd(S_L, M);
    chol_solve(S_L, M, rhs_lambda, u_lambda);
    // u_y = H^{-1}(bar_y - A^T u_lambda)
    std::vector<float> tmp(N, 0.0f);
    for (std::size_t i=0;i<N;++i) {
      float s = o->grad[i];
      for (std::size_t j=0;j<M;++j) s -= Aval[j*N + i] * u_lambda[j];
      tmp[i] = s;
    }
    std::vector<float> u_y;
    solve_Hinv_times(H_L, N, tmp, u_y);

    // Parameter grads (VJP rules)
    if (qn && qn->requires_grad) {
      // dL/dq = -u_y
      for (std::size_t i=0;i<N;++i) qn->grad[i] += -u_y[i];
    }
    if (Hn && Hn->requires_grad) {
      // Correct adjoint: dL/dH = -0.5 * (u_y * y_star^T + y_star * u_y^T)
      for (std::size_t i=0;i<N;++i) {
        for (std::size_t j=0;j<N;++j) {
          float g = -0.5f * (u_y[i]*y_star[j] + y_star[i]*u_y[j]);
          Hn->grad[i*N + j] += g;
        }
      }
    }
    if (bn && bn->requires_grad) {
      // dL/db = u_lambda
      for (std::size_t i=0;i<M;++i) bn->grad[i] +=  u_lambda[i];
    }
    if (An && An->requires_grad) {
      // Correct adjoint: dL/dA = -(u_lambda * y_star^T + lambda_star * u_y^T)
      for (std::size_t i=0;i<M;++i) {
        for (std::size_t j=0;j<N;++j) {
          An->grad[i*N + j] += -(u_lambda[i]*y_star[j] + lambda_star[i]*u_y[j]);
        }
      }
    }
  };

  return make_from_node(out);
}

} // namespace ag
