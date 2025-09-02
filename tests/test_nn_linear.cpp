// tests/test_nn_linear.cpp  (updated)
#include "test_framework.hpp"
#include "ag/nn/layers/linear.hpp"
#include "ag/core/variables.hpp"
#include <vector>
#include <numeric>
#include <cmath>

using ag::Variable;
using ag::nn::Linear;

static std::vector<double> ones(std::size_t n) { return std::vector<double>(n, 1.0); }
static std::vector<double> zeros(std::size_t n) { return std::vector<double>(n, 0.0); }

TEST("nn/linear/shape_and_params") {
    const std::size_t In = 3, Out = 2, B = 4;
    Linear lin(In, Out, /*bias=*/true, /*init_scale=*/0.01, /*seed=*/12345ull);

    // Parameter enumeration
    auto params = lin.parameters();
    ASSERT_TRUE(params.size() == 2); // W and b
    ASSERT_TRUE(params[0] != nullptr && params[1] != nullptr);
    ASSERT_TRUE(params[0]->value().size() == In*Out);
    ASSERT_TRUE(params[1]->value().size() == Out);

    // Forward shape
    Variable x(zeros(B*In), {B, In}, /*requires_grad=*/true);
    auto y = lin.forward(x);
    ASSERT_TRUE(y.value().size() == B*Out);
}

TEST("nn/linear/forward_bias_broadcast") {
    const std::size_t In = 4, Out = 3, B = 5;
    Linear lin(In, Out, /*bias=*/true, /*init_scale=*/0.02, /*seed=*/7ull);

    auto params = lin.parameters();
    Variable& W = *params[0];
    Variable& b = *params[1];

    // x = 0 -> y should equal row-wise broadcast of b
    Variable x(zeros(B*In), {B, In}, /*requires_grad=*/true);
    auto y = lin.forward(x);

    const auto& yv = y.value();
    const auto& bv = b.value();
    for (std::size_t i = 0; i < B; ++i) {
        for (std::size_t j = 0; j < Out; ++j) {
            ASSERT_NEAR(yv[i*Out + j], bv[j], 1e-12);
        }
    }

    // Backward with seed of ones: dW should be 0, db should be B
    y.backward(ones(B*Out));
    for (double gw : W.grad()) ASSERT_NEAR(gw, 0.0, 1e-12);
    for (double gb : b.grad()) ASSERT_NEAR(gb, double(B), 1e-12);
}

TEST("nn/linear/backward_wrt_x_sum_loss") {
    const std::size_t In = 3, Out = 4, B = 2;
    Linear lin(In, Out, /*bias=*/true, /*init_scale=*/0.05, /*seed=*/123ull);

    // x with known values
    std::vector<double> xv(B*In);
    std::iota(xv.begin(), xv.end(), 1.0); // 1,2,3,4,5,6
    Variable x(xv, {B, In}, /*requires_grad=*/true);

    // Forward
    auto y = lin.forward(x);

    // Loss = sum(y)  -> seed is all-ones
    const std::vector<double> seed = ones(B*Out);
    y.backward(seed);

    // Analytic d(sum(y))/dx = 1_row @ W^T = sum over columns of W, same for each batch row
    auto params = lin.parameters();
    const auto& Wv = params[0]->value(); // shape [In,Out] row-major
    std::vector<double> expected_dx(In, 0.0);
    for (std::size_t i = 0; i < In; ++i) {
        double s = 0.0;
        for (std::size_t j = 0; j < Out; ++j) s += Wv[i*Out + j];
        expected_dx[i] = s;
    }

    const auto& gx = x.grad();
    ASSERT_TRUE(gx.size() == B*In);
    for (std::size_t b = 0; b < B; ++b) {
        for (std::size_t i = 0; i < In; ++i) {
            ASSERT_NEAR(gx[b*In + i], expected_dx[i], 1e-10);
        }
    }
}

TEST("nn/linear/recompute_equals_after_zero") {
    const std::size_t In = 2, Out = 2, B = 3;
    Linear lin(In, Out, /*bias=*/true, /*init_scale=*/0.01, /*seed=*/99ull);
    auto params = lin.parameters();
    Variable& W = *params[0];
    Variable& BIAS = *params[1];

    // Input
    std::vector<double> xv(B*In);
    for (std::size_t i = 0; i < xv.size(); ++i) xv[i] = double(i+1);
    Variable x(xv, {B, In}, /*requires_grad=*/true);

    auto y = lin.forward(x);
    const std::vector<double> seed = ones(B*Out);

    // First backward
    y.backward(seed);
    auto gx1 = x.grad();
    auto gW1 = W.grad();
    auto gb1 = BIAS.grad();

    // Clear grads across graph & params
    x.zero_grad();
    W.zero_grad();
    BIAS.zero_grad();
    y.zero_grad();

    // Second backward should reproduce the same grads
    y.backward(seed);
    auto gx2 = x.grad();
    auto gW2 = W.grad();
    auto gb2 = BIAS.grad();

    ASSERT_TRUE(gx1.size() == gx2.size());
    ASSERT_TRUE(gW1.size() == gW2.size());
    ASSERT_TRUE(gb1.size() == gb2.size());

    for (std::size_t i = 0; i < gx1.size(); ++i) ASSERT_NEAR(gx2[i], gx1[i], 1e-12);
    for (std::size_t i = 0; i < gW1.size(); ++i) ASSERT_NEAR(gW2[i], gW1[i], 1e-12);
    for (std::size_t i = 0; i < gb1.size(); ++i) ASSERT_NEAR(gb2[i], gb1[i], 1e-12);
}
