
// tests/test_variable_basic.cpp
#include "test_framework.hpp"
#include <vector>
#include <cmath>
#include <iostream>

#include "ag/core/variables.hpp"
#include "ag/core/tensor_utils.hpp"

using ag::Variable;
using ag::add;
using ag::sub;
using ag::mul;
using ag::div;
using ag::neg;
using ag::sinv;
using ag::cosv;
using ag::expv;
using ag::matmul;

static Variable scalar(double v, bool requires_grad=true) {
    return Variable(std::vector<double>{v}, std::vector<std::size_t>{1}, requires_grad);
}

static Variable tensor(const std::vector<double>& vals, const std::vector<std::size_t>& shape, bool requires_grad=true) {
    return Variable(vals, shape, requires_grad);
}

static double sum_values(const Variable& v) {
    double s = 0.0;
    for (double x : v.value()) s += x;
    return s;
}

TEST("variable/basic_forward_backward_scalar") {
    // y = x^2 + 3x + 1
    double x = 1.7;
    auto vx = scalar(x, /*requires_grad=*/true);
    auto y = add(add(mul(vx, vx), mul(scalar(3.0,false), vx)), scalar(1.0,false));
    // scalar output; call backward() with default seed=1
    y.backward();
    ASSERT_NEAR(y.value()[0], x*x + 3*x + 1.0, 1e-12);
    ASSERT_NEAR(vx.grad()[0], 2*x + 3.0, 1e-8);
}

TEST("variable/finite_diff_vector") {
    // f(x) = sum_i sin x_i * cos(2 x_i)
    std::vector<double> x = { -0.3, 0.2, 1.1, -2.3 };
    Variable y = scalar(0.0, /*requires_grad=*/true);
    std::vector<Variable> v;
    v.reserve(x.size());
    for (double xi : x) v.push_back(scalar(xi, /*requires_grad=*/true));
    for (auto& xi : v) {
        y = add(y, mul(sinv(xi), cosv(mul(scalar(2.0,false), xi))));
    }
    // Backward: scalar output; seed 1
    y.backward();
    for (size_t i = 0; i < x.size(); ++i) {
        double g = std::cos(x[i]) * std::cos(2*x[i]) - 2.0 * std::sin(x[i]) * std::sin(2*x[i]);
        ASSERT_NEAR(v[i].grad()[0], g, 1e-5);
    }
}

TEST("variable/broadcast_add") {
    // A:[3,1], B:[1,4] -> C:[3,4], loss=sum(C) -> grads accumulate correctly
    Variable A = tensor({1,2,3}, {3,1}, /*requires_grad=*/true);
    Variable B = tensor({10,20,30,40}, {1,4}, /*requires_grad=*/true);
    Variable C = add(A,B);
    // Seed gradient with ones to emulate loss=sum(C)
    std::vector<double> seed(C.value().size(), 1.0);
    C.backward(seed);
    // dL/dA elements should each be 4 (one per broadcast col)
    for (size_t i=0;i<3;i++) ASSERT_NEAR(A.grad()[i], 4.0, 1e-12);
    // dL/dB elements should each be 3 (one per broadcast row)
    for (size_t j=0;j<4;j++) ASSERT_NEAR(B.grad()[j], 3.0, 1e-12);
}

// --- Added tests: gradient accumulation, zero_grad, pow/exp chain, div numeric check ---

TEST("variable/grad_accumulation_and_zero") {
    // y = sum(a + a) --> dL/da = 2 * ones
    Variable a = tensor({1.0, -2.0, 3.5}, {3}, /*requires_grad=*/true);
    Variable y = add(a, a);
    std::vector<double> seed(y.value().size(), 1.0); // emulate sum(C)
    y.backward(seed);
    ASSERT_NEAR(a.grad()[0], 2.0, 1e-12);
    ASSERT_NEAR(a.grad()[1], 2.0, 1e-12);
    ASSERT_NEAR(a.grad()[2], 2.0, 1e-12);

    // zero the grad and check cleared
    a.zero_grad();
    for (double g : a.grad()) ASSERT_NEAR(g, 0.0, 1e-18);
}

TEST("variable/pow_and_exp_chain_rule") {
    // y_i = exp( (x_i)^3 ) ; dy/dx = 3 x^2 * exp(x^3)
    Variable x = tensor({-1.2, 0.5, 2.0}, {3}, /*requires_grad=*/true);
    Variable three = scalar(3.0, /*requires_grad=*/false);
    Variable x3 = pow(x, three);
    Variable y  = expv(x3);
    std::vector<double> seed(y.value().size(), 1.0); // emulate sum(y)
    y.backward(seed);
    std::vector<double> xv = x.value();
    for (size_t i=0;i<xv.size();++i) {
        double expect = 3.0 * xv[i]*xv[i] * std::exp(xv[i]*xv[i]*xv[i]);
        ASSERT_NEAR(x.grad()[i], expect, 1e-4);
    }
}

TEST("variable/div_numeric_grad") {
    // f(x) = sum( x / c ) with c!=0 -> grad = 1/c
    Variable x = tensor({0.7, -1.1, 2.4}, {3}, /*requires_grad=*/true);
    Variable c = tensor({2.0, 2.0, 2.0}, {3}, /*requires_grad=*/false);
    Variable y = div(x, c);
    std::vector<double> seed(y.value().size(), 1.0);
    y.backward(seed);
    for (double g : x.grad()) ASSERT_NEAR(g, 0.5, 1e-12);
    // (finite-diff unnecessary here; analytic is exact for constant divisor)
}

TEST("variable/no_grad_operand_does_not_accumulate") {
    // If b does not require grad, its grad should remain zero
    Variable a = tensor({1.0, 2.0}, {2}, /*requires_grad=*/true);
    Variable b = tensor({3.0, 4.0}, {2}, /*requires_grad=*/false);
    Variable y = mul(a, b); // y = a * const
    std::vector<double> seed(y.value().size(), 1.0);
    y.backward(seed);
    // dL/da = b ; dL/db = 0
    ASSERT_NEAR(a.grad()[0], 3.0, 1e-12);
    ASSERT_NEAR(a.grad()[1], 4.0, 1e-12);
    bool any_nonzero = false;
    for (double g : b.grad()) any_nonzero = any_nonzero || (std::fabs(g) > 1e-20);
    ASSERT_TRUE(!any_nonzero);
}

