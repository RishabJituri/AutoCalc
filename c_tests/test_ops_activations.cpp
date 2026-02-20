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

static std::vector<float> ones(std::size_t n){ return std::vector<float>(n,1.0f); }

TEST("ops/activations/relu_forward_backward") {
    std::vector<float> xv = {-2.0f, -1.0f, 0.0f, 1.5f, 3.0f};
    Variable x(xv, {5}, /*requires_grad=*/true);
    auto y = relu(x);

    ASSERT_TRUE(y.value().size()==5);
    ASSERT_NEAR(y.value()[0], 0.0f, 1e-6f);
    ASSERT_NEAR(y.value()[1], 0.0f, 1e-6f);
    ASSERT_NEAR(y.value()[2], 0.0f, 1e-6f);
    ASSERT_NEAR(y.value()[3], 1.5f, 1e-6f);
    ASSERT_NEAR(y.value()[4], 3.0f, 1e-6f);

    y.backward(ones(5));
    const auto& gx = x.grad();
    ASSERT_NEAR(gx[0], 0.0f, 1e-6f);
    ASSERT_NEAR(gx[1], 0.0f, 1e-6f);
    ASSERT_NEAR(gx[2], 0.0f, 1e-6f);
    ASSERT_NEAR(gx[3], 1.0f, 1e-6f);
    ASSERT_NEAR(gx[4], 1.0f, 1e-6f);
}

TEST("ops/activations/sigmoid_numeric_grad") {
    std::vector<float> xv = { -0.3f, 0.0f, 0.7f };
    Variable x(xv, {3}, /*requires_grad=*/true);
    auto y = sigmoid(x);
    y.backward(ones(3));
    const auto gx = x.grad();

    const float eps=1e-4f;
    for (std::size_t i=0;i<xv.size();++i) {
        auto xp = xv; xp[i]+=eps;
        auto xm = xv; xm[i]-=eps;
        Variable Xp(xp, {3}, false), Xm(xm, {3}, false);
        auto yp = sigmoid(Xp), ym = sigmoid(Xm);
        float sp=0.0f, sm=0.0f; for (float v:yp.value()) sp+=v; for (float v:ym.value()) sm+=v;
        float gnum = (sp-sm)/(2*eps);
        ASSERT_NEAR(gx[i], gnum, 1e-3f);
    }
}

TEST("ops/activations/tanh_numeric_grad") {
    std::vector<float> xv = { -1.2f, 0.2f, 1.1f };
    Variable x(xv, {3}, true);
    auto y = tanhv(x);
    y.backward(ones(3));
    auto gx = x.grad();

    const float eps=1e-4f;
    for (std::size_t i=0;i<xv.size();++i) {
        auto xp=xv; xp[i]+=eps;
        auto xm=xv; xm[i]-=eps;
        Variable Xp(xp,{3},false), Xm(xm,{3},false);
        auto yp=tanhv(Xp), ym=tanhv(Xm);
        float sp=0.0f,sm=0.0f; for(float v:yp.value()) sp+=v; for(float v:ym.value()) sm+=v;
        ASSERT_NEAR(gx[i], (sp-sm)/(2*eps), 1e-3f);
    }
}

TEST("ops/activations/log_forward_grad") {
    std::vector<float> xv = { 0.5f, 1.0f, 2.0f };
    Variable x(xv, {3}, true);
    auto y = logv(x);
    y.backward(ones(3));
    auto gx = x.grad();
    ASSERT_NEAR(gx[0], 1.0f/0.5f, 1e-6f);
    ASSERT_NEAR(gx[1], 1.0f/1.0f, 1e-6f);
    ASSERT_NEAR(gx[2], 1.0f/2.0f, 1e-6f);
}

TEST("ops/activations/clamp_forward_backward") {
    std::vector<float> xv = { -2.0f, -0.5f, 0.2f, 0.9f, 2.3f };
    Variable x(xv, {5}, true);
    auto y = clamp(x, -0.5f, 1.0f);
    y.backward(ones(5));
    auto gx = x.grad();
    ASSERT_NEAR(gx[0], 0.0f, 1e-6f);
    ASSERT_NEAR(gx[1], 0.0f, 1e-6f);
    ASSERT_NEAR(gx[2], 1.0f, 1e-6f);
    ASSERT_NEAR(gx[3], 1.0f, 1e-6f);
    ASSERT_NEAR(gx[4], 0.0f, 1e-6f);
}
