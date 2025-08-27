// tests/test_qp_adjoint_and_fd.cpp
#include "test_framework.hpp"
#include "ag/core/implicit_qp.hpp"
#include "ag/core/variables.hpp"
#include <random>
#include <algorithm>
#include <numeric>
#include <limits>
#include <cmath>

#ifndef ASSERT_EQ
#define ASSERT_EQ(a,b) do { if (!((a)==(b))) throw std::runtime_error(std::string("ASSERT_EQ failed: " #a " != " #b)); } while(0)
#endif

using ag::Variable;
using ag::QPSolveEq;

static Variable tensor(const std::vector<double>& v,
                       const std::vector<std::size_t>& shape,
                       bool req=true) {
    return Variable(v, shape, req);
}

// ---------- Tiny dense helpers (row-major) ----------
static void cholesky_spd(std::vector<double>& M, std::size_t n) {
    for (std::size_t i=0;i<n;++i){
        for (std::size_t j=0;j<=i;++j){
            double s = M[i*n + j];
            for (std::size_t k=0;k<j;++k) s -= M[i*n+k]*M[j*n+k];
            if (i==j) {
                if (s <= 0.0) throw std::runtime_error("Cholesky failed");
                M[i*n+i] = std::sqrt(s);
            } else {
                M[i*n+j] = s / M[j*n+j];
            }
        }
        for (std::size_t j=i+1;j<n;++j) M[i*n+j] = 0.0;
    }
}
static void forward_subst(const std::vector<double>& L, std::size_t n,
                          const std::vector<double>& b, std::vector<double>& y) {
    y.assign(n,0.0);
    for (std::size_t i=0;i<n;++i) {
        double s=b[i];
        for (std::size_t k=0;k<i;++k) s -= L[i*n+k]*y[k];
        y[i] = s / L[i*n+i];
    }
}
static void back_subst(const std::vector<double>& L, std::size_t n,
                       const std::vector<double>& y, std::vector<double>& x) {
    x.assign(n,0.0);
    for (int ii=int(n)-1; ii>=0; --ii) {
        std::size_t i=std::size_t(ii);
        double s=y[i];
        for (std::size_t k=i+1;k<n;++k) s -= L[k*n+i]*x[k];
        x[i] = s / L[i*n+i];
    }
}
static void chol_solve(const std::vector<double>& L, std::size_t n,
                       const std::vector<double>& b, std::vector<double>& x) {
    std::vector<double> y;
    forward_subst(L,n,b,y);
    back_subst(L,n,y,x);
}
static std::vector<double> solve_Hinv_times(const std::vector<double>& L, std::size_t n,
                                            const std::vector<double>& v) {
    std::vector<double> x; chol_solve(L,n,v,x); return x;
}
static std::vector<double> matvec(const std::vector<double>& M, std::size_t r, std::size_t c,
                                  const std::vector<double>& x) {
    std::vector<double> y(r,0.0);
    for (std::size_t i=0;i<r;++i) {
        double s=0.0;
        for (std::size_t j=0;j<c;++j) s += M[i*c + j]*x[j];
        y[i]=s;
    }
    return y;
}
static double dot(const std::vector<double>& a, const std::vector<double>& b){
    double s=0.0; for (std::size_t i=0;i<a.size(); ++i) s += a[i]*b[i]; return s;
}
static double frob_inner(const std::vector<double>& A, const std::vector<double>& B){
    double s=0.0; for (std::size_t i=0;i<A.size(); ++i) s += A[i]*B[i]; return s;
}
static double norm2(const std::vector<double>& v) {
    double s=0.0; for (double x: v) s += x*x; return std::sqrt(s);
}
static std::mt19937_64& rng() {
    static std::mt19937_64 gen(0xC001D00DULL);
    return gen;
}
static double randn(double s=1.0) {
    static std::normal_distribution<double> N(0.0,1.0);
    return s*N(rng());
}
static double randu(double a, double b) {
    std::uniform_real_distribution<double> U(a,b);
    return U(rng());
}
static std::vector<double> make_spd(std::size_t n, double mu=1e-3) {
    std::vector<double> R(n*n);
    for (std::size_t i=0;i<n*n;++i) R[i] = randn();
    std::vector<double> H(n*n,0.0);
    for (std::size_t i=0;i<n;++i) {
        for (std::size_t j=0;j<n;++j) {
            double s=0.0;
            for (std::size_t k=0;k<n;++k) s += R[k*n+i]*R[k*n+j];
            H[i*n+j] = s + ((i==j)?mu:0.0);
        }
    }
    return H;
}
static std::vector<double> build_S(const std::vector<double>& HL, std::size_t n,
                                   const std::vector<double>& A, std::size_t m) {
    // S = A H^{-1} A^T via column-by-column
    std::vector<double> S(m*m,0.0), col(n), HinvATcol(n);
    for (std::size_t j=0;j<m;++j){
        for (std::size_t i=0;i<n;++i) col[i] = A[j*n + i]; // A^T e_j
        HinvATcol = solve_Hinv_times(HL,n,col);
        for (std::size_t i=0;i<m;++i){
            double s=0.0; for (std::size_t k=0;k<n;++k) s += A[i*n+k]*HinvATcol[k];
            S[i*m + j] = s;
        }
    }
    return S;
}
static std::vector<double> solve_lambda(const std::vector<double>& H,
                                        const std::vector<double>& q,
                                        const std::vector<double>& A, std::size_t m, std::size_t n,
                                        const std::vector<double>& b) {
    std::vector<double> Hc = H; cholesky_spd(Hc, n);
    auto Hinv_q = solve_Hinv_times(Hc,n,q);
    std::vector<double> rhs(m,0.0);
    for (std::size_t i=0;i<m;++i){
        double s = -b[i];
        for (std::size_t k=0;k<n;++k) s -= A[i*n+k]*Hinv_q[k];
        rhs[i]=s;
    }
    std::vector<double> S = build_S(Hc,n,A,m);
    std::vector<double> SL = S; cholesky_spd(SL, m);
    std::vector<double> lam; chol_solve(SL,m,rhs,lam);
    return lam;
}
static void solve_adjoint(const std::vector<double>& H, const std::vector<double>& A,
                          std::size_t n, std::size_t m,
                          const std::vector<double>& g,
                          std::vector<double>& u, std::vector<double>& v) {
    // [H A^T; A 0] [u; v] = [g; 0]
    std::vector<double> HL = H; cholesky_spd(HL, n);
    // v = (A H^{-1} A^T)^{-1} (A H^{-1} g)
    auto Hinv_g = solve_Hinv_times(HL,n,g);
    std::vector<double> Ahinv_g(m,0.0);
    for (std::size_t i=0;i<m;++i){
        double s=0.0; for (std::size_t k=0;k<n;++k) s += A[i*n+k]*Hinv_g[k];
        Ahinv_g[i]=s;
    }
    std::vector<double> S = build_S(HL,n,A,m);
    std::vector<double> SL = S; cholesky_spd(SL, m);
    chol_solve(SL, m, Ahinv_g, v);
    // u = H^{-1}(g - A^T v)
    std::vector<double> g_minus_ATv(n,0.0);
    for (std::size_t i=0;i<n;++i){
        double s = g[i];
        for (std::size_t j=0;j<m;++j) s -= A[j*n + i]*v[j];
        g_minus_ATv[i]=s;
    }
    u = solve_Hinv_times(HL,n,g_minus_ATv);
}

// 0.5 * ||x||^2 loss utilities
static double loss_half_sum_sq(const Variable& x) {
    double L=0.0; for (double v: x.value()) L += 0.5*v*v; return L;
}
static void seed_from_half_sum_sq(Variable& x) {
    // d/dx (0.5 * sum x_i^2) = x
    x.backward(x.value());
}

// ----------- TESTS -----------

// Forward correctness sanity: M=0 closed-form; KKT residuals (looser eps = 1e-5)
TEST("implicit_qp/forward_sanity_unconstrained_and_kkt") {
    for (int t=0;t<6;++t){
        const std::size_t N = (t%3==0)?2: (t%3==1?3:5);
        const std::size_t M = (t<3)?0:1;

        std::vector<double> H = make_spd(N, 1e-3);
        std::vector<double> q(N); for (auto& v: q) v = randn();
        std::vector<double> A(M*N); for (auto& v: A) v = randn();
        std::vector<double> b(M); for (auto& v: b) v = randn();

        Variable Hv = tensor(H,{N,N});
        Variable qv = tensor(q,{N});
        Variable Av = tensor(A,{M,N});
        Variable bv = tensor(b,{M});

        Variable x = QPSolveEq(Hv,qv,Av,bv);

        if (M==0){
            // x* = -H^{-1} q
            std::vector<double> Hc = H; cholesky_spd(Hc,N);
            std::vector<double> mnegq = q; for (auto& z: mnegq) z = -z;
            std::vector<double> x_cf; chol_solve(Hc,N,mnegq,x_cf);
            for (std::size_t i=0;i<N;++i) ASSERT_NEAR(x.value()[i], x_cf[i], 1e-5);
        } else {
            // Check KKT residual
            auto lam = solve_lambda(H,q,A,M,N,b);
            auto Hx  = matvec(H,N,N,x.value());
            std::vector<double> kkt(N,0.0);
            for (std::size_t i=0;i<N;++i) kkt[i] = Hx[i] + q[i];
            for (std::size_t j=0;j<M;++j){
                for (std::size_t i=0;i<N;++i) kkt[i] += A[j*N + i]*lam[j];
            }
            ASSERT_NEAR(norm2(kkt), 0.0, 1e-5);

            auto Ax = matvec(A,M,N,x.value());
            double rpri=0.0; for (std::size_t i=0;i<M;++i) rpri=std::max(rpri,std::fabs(Ax[i]-b[i]));
            ASSERT_NEAR(rpri, 0.0, 1e-6);
        }
    }
}

// Adjoint identities: grad_q = -u, grad_b = v, grad_H = -sym(u x^T), grad_A = -(λ u^T + v x^T)
TEST("implicit_qp/adjoint_identities_match_backward") {
    std::vector<std::pair<int,int>> cases = {{2,0},{3,0},{3,1},{5,1},{5,2}};
    for (auto [Ni,Mi] : cases){
        const std::size_t N = static_cast<std::size_t>(Ni);
        const std::size_t M = static_cast<std::size_t>(Mi);

        // Random instance
        std::vector<double> H = make_spd(N, 2e-3); // tiny ridge
        std::vector<double> q(N); for (auto& v: q) v = randn();
        std::vector<double> A(M*N); for (auto& v: A) v = randn();
        std::vector<double> b(M); for (auto& v: b) v = randn();

        // Variables (requires_grad=true by default)
        Variable Hv = tensor(H,{N,N});
        Variable qv = tensor(q,{N});
        Variable Av = tensor(A,{M,N});
        Variable bv = tensor(b,{M});

        Variable x = QPSolveEq(Hv,qv,Av,bv);
        seed_from_half_sum_sq(x);         // g = x

        // Solve adjoint & lambda
        std::vector<double> u, v; solve_adjoint(H,A,N,M, x.value(), u, v);
        std::vector<double> lam = (M>0)? solve_lambda(H,q,A,M,N,b) : std::vector<double>();

        // grad_q
        ASSERT_EQ(qv.grad().size(), N);
        for (std::size_t i=0;i<N;++i) ASSERT_NEAR(qv.grad()[i], -u[i], 2e-6);

        // grad_b
        ASSERT_EQ(bv.grad().size(), M);
        for (std::size_t i=0;i<M;++i) ASSERT_NEAR(bv.grad()[i],  v[i], 2e-6);

        // grad_H = -0.5*(u x^T + x u^T)
        ASSERT_EQ(Hv.grad().size(), N*N);
        for (std::size_t i=0;i<N;++i){
            for (std::size_t j=0;j<N;++j){
                double ref = -0.5*(u[i]*x.value()[j] + x.value()[i]*u[j]);
                ASSERT_NEAR(Hv.grad()[i*N+j], ref, 5e-5);
            }
        }

        // grad_A = -(λ u^T + v x^T)  (skip if M==0)
        if (M>0){
            ASSERT_EQ(Av.grad().size(), M*N);
            for (std::size_t i=0;i<M;++i){
                for (std::size_t j=0;j<N;++j){
                    double ref = -(lam[i]*u[j] + v[i]*x.value()[j]);
                    ASSERT_NEAR(Av.grad()[i*N + j], ref, 5e-5);
                }
            }
        }
    }
}

// Directional FD vs. inner product of grads (hits H,A,q,b at once)
TEST("implicit_qp/directional_fd_matches_grad_inner_product") {
    std::vector<std::pair<int,int>> cases = {{3,0},{3,1},{5,2}};
    for (auto [Ni,Mi] : cases){
        const std::size_t N = static_cast<std::size_t>(Ni);
        const std::size_t M = static_cast<std::size_t>(Mi);

        std::vector<double> H = make_spd(N, 1e-3);
        std::vector<double> q(N); for (auto& v: q) v = randn();
        std::vector<double> A(M*N); for (auto& v: A) v = randn();
        std::vector<double> b(M); for (auto& v: b) v = randn();

        // Build random directions dH (sym), dA, dq, db with small magnitude
        std::vector<double> dH(N*N,0.0), dA(M*N,0.0), dq(N,0.0), db(M,0.0);
        for (std::size_t i=0;i<N;++i){
            for (std::size_t j=i;j<N;++j){
                double r = 1e-3*randn(); // small
                dH[i*N+j]=r; dH[j*N+i]=r;
            }
        }
        for (auto& z: dA) z = 1e-3*randn();
        for (auto& z: dq) z = 1e-3*randn();
        for (auto& z: db) z = 1e-3*randn();

        // Autodiff grads at basepoint with g = x (0.5*||x||^2)
        Variable Hv = tensor(H,{N,N});
        Variable qv = tensor(q,{N});
        Variable Av = tensor(A,{M,N});
        Variable bv = tensor(b,{M});
        Variable x0 = QPSolveEq(Hv,qv,Av,bv);
        seed_from_half_sum_sq(x0);

        // Inner product <grad, dθ>
        double inner = 0.0;
        inner += dot(qv.grad(), dq);
        inner += dot(bv.grad(), db);
        inner += frob_inner(Hv.grad(), dH);
        if (M>0) inner += frob_inner(Av.grad(), dA);

        // Central FD of L under combined perturbation
        auto Hpos = H, Hneg = H;
        auto Apos = A, Aneg = A;
        auto qpos = q, qneg = q;
        auto bpos = b, bneg = b;

        auto apply = [](std::vector<double>& dst, const std::vector<double>& delta, double alpha){
            for (std::size_t i=0;i<dst.size(); ++i) dst[i] += alpha*delta[i];
        };

        double eps = 1e-3;   // relatively small; we already scaled d* by 1e-3
        apply(Hpos,dH, eps); apply(Hneg,dH,-eps);
        apply(Apos,dA, eps); apply(Aneg,dA,-eps);
        apply(qpos,dq, eps); apply(qneg,dq,-eps);
        apply(bpos,db, eps); apply(bneg,db,-eps);

        Variable Hpv = tensor(Hpos,{N,N}, /*req=*/false);
        Variable Hnv = tensor(Hneg,{N,N}, /*req=*/false);
        Variable Apv = tensor(Apos,{M,N}, /*req=*/false);
        Variable Anv = tensor(Aneg,{M,N}, /*req=*/false);
        Variable qpv = tensor(qpos,{N},    /*req=*/false);
        Variable qnv = tensor(qneg,{N},    /*req=*/false);
        Variable bpv = tensor(bpos,{M},    /*req=*/false);
        Variable bnv = tensor(bneg,{M},    /*req=*/false);

        double Lpos = loss_half_sum_sq(QPSolveEq(Hpv,qpv,Apv,bpv));
        double Lneg = loss_half_sum_sq(QPSolveEq(Hnv,qnv,Anv,bnv));
        double fd = (Lpos - Lneg) / (2*eps);

        // Compare
        ASSERT_NEAR(inner, fd, 5e-4);
    }
}
