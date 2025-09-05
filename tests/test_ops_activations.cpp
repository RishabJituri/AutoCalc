#include "test_framework.hpp"
#include "ag/core/variables.hpp"
#include "ag/ops/activations.hpp"
#include <vector>
#include <cmath>
#include <memory>

using ag::Variable;
using ag::relu;
using ag::sigmoid;
using ag::tanhv;
using ag::logv;
using ag::clamp;

static std::vector<double> ones(std::size_t n){ return std::vector<double>(n,1.0); }

TEST("ops/activations/relu_forward_backward") {
    std::vector<double> xv = {-2.0, -1.0, 0.0, 1.5, 3.0};
    Variable x(xv, {5}, /*requires_grad=*/true);
    auto y = relu(x);

    ASSERT_TRUE(y.value().size()==5);
    ASSERT_NEAR(y.value()[0], 0.0, 1e-12);
    ASSERT_NEAR(y.value()[1], 0.0, 1e-12);
    ASSERT_NEAR(y.value()[2], 0.0, 1e-12);
    ASSERT_NEAR(y.value()[3], 1.5, 1e-12);
    ASSERT_NEAR(y.value()[4], 3.0, 1e-12);

    y.backward(ones(5));
    const auto& gx = x.grad();
    ASSERT_NEAR(gx[0], 0.0, 1e-12);
    ASSERT_NEAR(gx[1], 0.0, 1e-12);
    ASSERT_NEAR(gx[2], 0.0, 1e-12);
    ASSERT_NEAR(gx[3], 1.0, 1e-12);
    ASSERT_NEAR(gx[4], 1.0, 1e-12);
}

TEST("ops/activations/sigmoid_numeric_grad") {
    std::vector<double> xv = { -0.3, 0.0, 0.7 };
    Variable x(xv, {3}, /*requires_grad=*/true);
    auto y = sigmoid(x);
    y.backward(ones(3));
    const auto gx = x.grad();

    const double eps=1e-6;
    for (std::size_t i=0;i<xv.size();++i) {
        auto xp = xv; xp[i]+=eps;
        auto xm = xv; xm[i]-=eps;
        Variable Xp(xp, {3}, false), Xm(xm, {3}, false);
        auto yp = sigmoid(Xp), ym = sigmoid(Xm);
        double sp=0.0, sm=0.0; for (double v:yp.value()) sp+=v; for (double v:ym.value()) sm+=v;
        double gnum = (sp-sm)/(2*eps);
        ASSERT_NEAR(gx[i], gnum, 1e-6);
    }
}

TEST("ops/activations/tanh_numeric_grad") {
    std::vector<double> xv = { -1.2, 0.2, 1.1 };
    Variable x(xv, {3}, true);
    auto y = tanhv(x);
    y.backward(ones(3));
    auto gx = x.grad();

    const double eps=1e-6;
    for (std::size_t i=0;i<xv.size();++i) {
        auto xp=xv; xp[i]+=eps;
        auto xm=xv; xm[i]-=eps;
        Variable Xp(xp,{3},false), Xm(xm,{3},false);
        auto yp=tanhv(Xp), ym=tanhv(Xm);
        double sp=0.0,sm=0.0; for(double v:yp.value()) sp+=v; for(double v:ym.value()) sm+=v;
        ASSERT_NEAR(gx[i], (sp-sm)/(2*eps), 1e-6);
    }
}

TEST("ops/activations/log_forward_grad") {
    std::vector<double> xv = { 0.5, 1.0, 2.0 };
    Variable x(xv, {3}, true);
    auto y = logv(x);
    y.backward(ones(3));
    auto gx = x.grad();
    ASSERT_NEAR(gx[0], 1.0/0.5, 1e-12);
    ASSERT_NEAR(gx[1], 1.0/1.0, 1e-12);
    ASSERT_NEAR(gx[2], 1.0/2.0, 1e-12);
}

TEST("ops/activations/clamp_forward_backward") {
    std::vector<double> xv = { -2.0, -0.5, 0.2, 0.9, 2.3 };
    Variable x(xv, {5}, true);
    auto y = clamp(x, -0.5, 1.0);
    y.backward(ones(5));
    auto gx = x.grad();
    ASSERT_NEAR(gx[0], 0.0, 1e-12);
    ASSERT_NEAR(gx[1], 0.0, 1e-12);
    ASSERT_NEAR(gx[2], 1.0, 1e-12);
    ASSERT_NEAR(gx[3], 1.0, 1e-12);
    ASSERT_NEAR(gx[4], 0.0, 1e-12);
}
