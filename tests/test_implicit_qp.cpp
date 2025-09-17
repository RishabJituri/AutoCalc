// // tests/test_implicit_qp.cpp
// #include "test_framework.hpp"
// #include "ag/core/implicit_qp.hpp"
// #include "ag/core/variables.hpp"

// using ag::Variable;
// using ag::QPSolveEq;

// // helpers
// static Variable tensor(const std::vector<float>& v,
//                        const std::vector<std::size_t>& shape,
//                        bool req=true) {
//     return Variable(v, shape, req);
// }
// static float dot(const std::vector<float>& a, const std::vector<float>& b) {
//     float s=0; for (size_t i=0;i<a.size();++i) s += a[i]*b[i]; return s;
// }

// // H is 2x2 SPD, one equality constraint a^T x = b
// static void make_problem(Variable& H, Variable& q, Variable& A, Variable& b) {
//     H = tensor({ 2.0f, 0.3f,
//                  0.3f, 1.7f }, {2,2}, /*req=*/true);
//     q = tensor({ -1.0f, 0.2f }, {2}, /*req=*/true);
//     A = tensor({ 1.0f, 1.0f }, {1,2}, /*req=*/true);
//     b = tensor({ 1.0f }, {1}, /*req=*/true);
// }

// TEST("implicit_qp/feasibility_and_kkt_residual") {
//     Variable H,q,A,b; make_problem(H,q,A,b);
//     Variable x = QPSolveEq(H,q,A,b); // x*: shape [2]

//     // primal feasibility: a^T x == b
//     const auto xv = x.value();
//     const auto av = A.value();
//     const auto bv = b.value();
//     float ax = av[0]*xv[0] + av[1]*xv[1];
//     ASSERT_NEAR(ax, bv[0], 1e-8f);

//     // stationarity residual: r = Hx + q + A^T λ, for M=1 constraint
//     // recover λ from: λ = - (a·(Hx+q)) / (a·a)
//     std::vector<float> Hx(2);
//     const auto Hv = H.value();
//     Hx[0] = Hv[0]*xv[0] + Hv[1]*xv[1];
//     Hx[1] = Hv[2]*xv[0] + Hv[3]*xv[1];
//     std::vector<float> Hx_plus_q = {Hx[0] + q.value()[0], Hx[1] + q.value()[1]};
//     float aa = av[0]*av[0] + av[1]*av[1];
//     float lam = - dot(av, Hx_plus_q) / aa;
//     std::vector<float> r = { Hx_plus_q[0] + av[0]*lam,
//                               Hx_plus_q[1] + av[1]*lam };
//     float rn = std::sqrt(r[0]*r[0] + r[1]*r[1]);
//     ASSERT_NEAR(rn, 0.0f, 1e-6f);
// }

// TEST("implicit_qp/grad_wrt_q_finite_diff") {
//     // L(H,q,A,b) = sum(x*^2), x* = QPSolveEq(...)
//     // check dL/dq[0] by finite differences
//     Variable H,q,A,b; make_problem(H,q,A,b);
//     Variable x = QPSolveEq(H,q,A,b);

//     // autodiff: seed dL/dx = 2x
//     std::vector<float> seed( x.value().size() );
//     for (size_t i=0;i<seed.size();++i) seed[i] = 2.0f * x.value()[i];
//     x.backward(seed);
//     float grad_ad = q.grad()[0];

//     // finite diff on q[0]
//     float eps = 1e-5f;
//     auto qv = q.value();
//     auto qv_pos = qv; qv_pos[0] += eps;
//     auto qv_neg = qv; qv_neg[0] -= eps;

//     Variable qpos = tensor(qv_pos, {2}, /*req=*/false);
//     Variable qneg = tensor(qv_neg, {2}, /*req=*/false);

//     Variable x_pos = QPSolveEq(H, qpos, A, b);
//     Variable x_neg = QPSolveEq(H, qneg, A, b);

//     float Lpos=0, Lneg=0;
//     for (float v : x_pos.value()) Lpos += v*v;
//     for (float v : x_neg.value()) Lneg += v*v;

//     float grad_fd = (Lpos - Lneg) / (2*eps);
//     ASSERT_NEAR(grad_ad, grad_fd, 1e-3f);
// }

// TEST("implicit_qp/grad_wrt_b_finite_diff") {
//     // same loss, check gradient w.r.t. b[0]
//     Variable H,q,A,b; make_problem(H,q,A,b);
//     Variable x = QPSolveEq(H,q,A,b);
//     std::vector<float> seed( x.value().size() );
//     for (size_t i=0;i<seed.size();++i) seed[i] = 2.0f * x.value()[i];
//     x.backward(seed);
//     float grad_ad = b.grad()[0];

//     float eps = 1e-5f;
//     auto bv = b.value();
//     auto bv_pos = bv; bv_pos[0] += eps;
//     auto bv_neg = bv; bv_neg[0] -= eps;

//     Variable bpos = tensor(bv_pos, {1}, /*req=*/false);
//     Variable bneg = tensor(bv_neg, {1}, /*req=*/false);

//     Variable x_pos = QPSolveEq(H, q, A, bpos);
//     Variable x_neg = QPSolveEq(H, q, A, bneg);

//     float Lpos=0, Lneg=0;
//     for (float v : x_pos.value()) Lpos += v*v;
//     for (float v : x_neg.value()) Lneg += v*v;

//     float grad_fd = (Lpos - Lneg) / (2*eps);
//     ASSERT_NEAR(grad_ad, grad_fd, 1e-3f);
// }
