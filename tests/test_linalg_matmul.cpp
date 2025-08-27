// tests/test_linalg_matmul.cpp
#include "test_framework.hpp"
#include <vector>
#include <cmath>
#include <iostream>

#include "ag/core/variables.hpp"
#include "ag/core/tensor_utils.hpp"

using ag::Variable;
using ag::matmul;
using ag::mul;

static Variable tensor(const std::vector<double>& vals, const std::vector<std::size_t>& shape, bool requires_grad=true) {
    return Variable(vals, shape, requires_grad);
}

// --- existing tests ---

TEST("linalg/matmul_shapes_and_backward") {
    // A:[2,3], B:[3,4] -> C:[2,4]
    Variable A = tensor({0,0,0, 0,0,0}, {2,3}, /*requires_grad=*/true);
    Variable B = tensor({1,1,1,1, 1,1,1,1, 1,1,1,1}, {3,4}, /*requires_grad=*/true);
    Variable C = matmul(A,B);
    auto shp = C.shape();
    ASSERT_TRUE(shp.size()==2 && shp[0]==2 && shp[1]==4);
    // loss = sum(C); emulate by seeding ones
    std::vector<double> seed(C.value().size(), 1.0);
    C.backward(seed);
    ASSERT_TRUE(A.grad().size()==6);
    ASSERT_TRUE(B.grad().size()==12);
}

TEST("linalg/matmul_numeric_grad_single_entry") {
    // loss = sum((A@B)^2) ; check grad w.r.t. A[0,0]
    Variable A = tensor({0.5,-0.3,0.1, 0.2,-0.7,1.1}, {2,3}, /*requires_grad=*/true);
    Variable B = tensor({0.9, -0.4, 0.3, 1.2,
                         -0.1, 0.5, -0.6, 0.8,
                         0.2, 0.3, -0.7, -0.9}, {3,4}, /*requires_grad=*/true);
    auto C = matmul(A,B);
    // loss L = sum(C^2) -> dL/dC = 2*C
    std::vector<double> seed(C.value().size());
    for (size_t i=0;i<seed.size();++i) seed[i] = 2.0 * C.value()[i];
    C.backward(seed);
    // finite difference on A[0,0]
    double eps = 1e-5;
    auto Avals = A.value();
    auto Apos = tensor(Avals, {2,3}, /*requires_grad=*/false);
    Avals[0] += eps;
    Apos = tensor(Avals, {2,3}, false);
    auto Cpos = matmul(Apos, B);
    double Lpos = 0.0; for (double v : Cpos.value()) Lpos += v*v;

    Avals = A.value();
    Avals[0] -= eps;
    auto Aneg = tensor(Avals, {2,3}, false);
    auto Cneg = matmul(Aneg, B);
    double Lneg = 0.0; for (double v : Cneg.value()) Lneg += v*v;

    double fd = (Lpos - Lneg) / (2*eps);
    double an = A.grad()[0];
    ASSERT_NEAR(an, fd, 1e-3);
}

// --- new tests appended below ---

TEST("linalg/matmul_associativity_numeric") {
    // Numerically check (A@B)@C ≈ A@(B@C) for small matrices
    Variable A = tensor({ 0.5,  1.0,
                         -0.3,  0.2}, {2,2}, /*requires_grad=*/false);
    Variable B = tensor({ 1.1, -0.4,
                          0.3,  0.7}, {2,2}, /*requires_grad=*/false);
    Variable C = tensor({ 0.9,  0.0,
                          0.2,  1.3}, {2,2}, /*requires_grad=*/false);

    auto L = matmul(matmul(A,B), C);
    auto R = matmul(A, matmul(B,C));
    for (size_t i=0;i<L.value().size();++i) {
        ASSERT_NEAR(L.value()[i], R.value()[i], 1e-12);
    }
}

TEST("linalg/matmul_grad_wrt_B_numeric") {
    // L = sum((A@B)^2) ; check gradient for B[1,0]
    Variable A = tensor({ 0.2, -0.1, 0.3,
                         -0.5,  0.7, 0.8}, {2,3}, /*requires_grad=*/true);
    Variable B = tensor({-0.6, 0.9,
                          0.4, 0.1,
                          0.0, 0.5}, {3,2}, /*requires_grad=*/true);

    auto C = matmul(A,B);
    // dL/dC = 2*C for L = sum(C^2)
    std::vector<double> seed(C.value().size());
    for (size_t i=0;i<seed.size();++i) seed[i] = 2.0 * C.value()[i];
    C.backward(seed);

    // Finite difference on B[1,0] (linear index 2 in row-major for shape {3,2})
    double eps = 1e-5;
    auto Bv = B.value();
    auto Bv_pos = Bv; Bv_pos[2] += eps;
    auto Bv_neg = Bv; Bv_neg[2] -= eps;

    auto Bpos = tensor(Bv_pos, {3,2}, /*requires_grad=*/false);
    auto Bneg = tensor(Bv_neg, {3,2}, /*requires_grad=*/false);

    auto Cpos = matmul(A, Bpos);
    auto Cneg = matmul(A, Bneg);

    double Lpos=0, Lneg=0;
    for (double v : Cpos.value()) Lpos += v*v;
    for (double v : Cneg.value()) Lneg += v*v;

    double fd = (Lpos - Lneg)/(2*eps);
    ASSERT_NEAR(B.grad()[2], fd, 1e-3);
}

TEST("linalg/matmul_with_identity_keeps_matrix") {
    // Check A @ I = A and I @ A = A (where dimensions match)
    Variable A = tensor({ 1,2,3,
                          4,5,6}, {2,3}, /*requires_grad=*/false);

    // Right-identity: A(2x3) @ I3(3x3) -> (2x3) equals A
    Variable I3 = tensor({1,0,0,
                          0,1,0,
                          0,0,1}, {3,3}, /*requires_grad=*/false);
    auto R1 = matmul(A, I3);
    ASSERT_TRUE(R1.shape()==std::vector<std::size_t>({2,3}));
    for (size_t i=0;i<A.value().size();++i) {
        ASSERT_NEAR(R1.value()[i], A.value()[i], 1e-12);
    }

    // Left-identity: I2(2x2) @ A2(2x2) -> equals A2
    Variable I2 = tensor({1,0,
                          0,1}, {2,2}, /*requires_grad=*/false);
    Variable A2 = tensor({3,4,
                          5,6}, {2,2}, /*requires_grad=*/false);
    auto R2 = matmul(I2, A2);
    for (size_t i=0;i<R2.value().size();++i) {
        ASSERT_NEAR(R2.value()[i], A2.value()[i], 1e-12);
    }
}

// --- Added tests: associativity (numeric), grad w.r.t. B numeric check, identity behavior ---

static Variable eye2(bool req=true) {
    return tensor({1,0,0,1}, {2,2}, req);
}

TEST("linalg/matmul_associativity_numeric") {
    // (A@B)@C ≈ A@(B@C)  (small numeric sanity)
    Variable A = tensor({0.5, 1.0,
                         -0.3, 0.2}, {2,2}, false);
    Variable B = tensor({1.1, -0.4,
                          0.3,  0.7}, {2,2}, false);
    Variable C = tensor({0.9, 0.0,
                         0.2, 1.3}, {2,2}, false);
    auto L = matmul(matmul(A,B), C);
    auto R = matmul(A, matmul(B,C));
    for (size_t i=0;i<L.value().size();++i) {
        ASSERT_NEAR(L.value()[i], R.value()[i], 1e-12);
    }
}

TEST("linalg/matmul_grad_wrt_B_numeric") {
    // L = sum((A@B)^2) ; finite-diff check for B[1,0]
    Variable A = tensor({ 0.2, -0.1, 0.3,
                         -0.5,  0.7, 0.8}, {2,3}, true);
    Variable B = tensor({-0.6, 0.9,
                          0.4, 0.1,
                          0.0, 0.5}, {3,2}, true);
    auto C = matmul(A,B);
    std::vector<double> seed(C.value().size());
    for (size_t i=0;i<seed.size();++i) seed[i] = 2.0 * C.value()[i]; // d/dC sum(C^2) = 2C
    C.backward(seed);

    double eps = 1e-5;
    auto Bv = B.value();
    auto Bv_pos = Bv; Bv_pos[2] += eps; // [1,0] in row-major index = 1*2 + 0 = 2
    auto Bv_neg = Bv; Bv_neg[2] -= eps;

    auto Bpos = tensor(Bv_pos, {3,2}, false);
    auto Bneg = tensor(Bv_neg, {3,2}, false);
    auto Cpos = matmul(A, Bpos);
    auto Cneg = matmul(A, Bneg);
    double Lpos=0, Lneg=0;
    for (double v : Cpos.value()) Lpos += v*v;
    for (double v : Cneg.value()) Lneg += v*v;
    double fd = (Lpos - Lneg)/(2*eps);

    ASSERT_NEAR(B.grad()[2], fd, 1e-3);
}

TEST("linalg/matmul_with_identity_keeps_matrix") {
    // A @ I = A, I @ A = A
    Variable A = tensor({ 1,2,3,
                          4,5,6}, {2,3}, false);
    Variable I3 = tensor({1,0,0,
                          0,1,0,
                          0,0,1}, {3,3}, false);
    auto L = matmul(A, I3);
    ASSERT_TRUE(L.shape()==std::vector<std::size_t>({2,3}));
    for (size_t i=0;i<A.value().size();++i) ASSERT_NEAR(L.value()[i], A.value()[i], 1e-12);

    Variable I2 = eye2(false);
    Variable A2 = eye2(false);
    auto R = matmul(I2, A2);
    for (size_t i=0;i<R.value().size();++i) ASSERT_NEAR(R.value()[i], A2.value()[i], 1e-12);
}

