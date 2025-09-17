// ============================
// File: tests/test_optim_sgd.cpp
// ============================
#include "test_framework.hpp"
#include "ag/nn/layers/linear.hpp"
#include "ag/nn/optim/sgd.hpp"
#include "ag/core/variables.hpp"
#include "ag/ops/elementwise.hpp"
#include "ag/ops/reduce.hpp"
#include <vector>
#include <cmath>

using ag::Variable;
using ag::nn::Linear;
using ag::nn::SGD;
using ag::reduce_mean;
using ag::sub;
using ag::mul;

static std::vector<float> zeros(std::size_t n) { return std::vector<float>(n, 0.0f); }

TEST("optim/sgd/linear_regression_decreases_loss") {
    const std::size_t In=2, Out=1, B=16;
    // True weights/bias to learn
    const float w0 = 2.0f, w1 = -3.0f, btrue = 0.5f;
 
    // Model
    Linear lin(In, Out, /*bias=*/true, /*init_scale=*/0.01f, /*seed=*/123ull);
    SGD opt(/*lr=*/0.1f, /*momentum=*/0.9f, /*nesterov=*/true, /*weight_decay=*/0.0f);

    // One mini-batch synthetic data
    std::vector<float> xv(B*In);
    for (std::size_t i=0;i<B;++i) {
        float x0 = float(i) / float(B);
        float x1 = 1.0f - x0;
        xv[2*i + 0] = x0;
        xv[2*i + 1] = x1;
    }
    Variable X(xv, {B,In}, /*requires_grad=*/true);

    // Targets: y = w0*x0 + w1*x1 + b
    std::vector<float> yv(B*Out);
    for (std::size_t i=0;i<B;++i) {
        float x0 = xv[2*i+0], x1 = xv[2*i+1];
        yv[i] = w0*x0 + w1*x1 + btrue;
    }
    Variable Y(yv, {B,Out}, /*requires_grad=*/false);

    // Compute initial loss
    auto Yhat0 = lin.forward(X);
    auto L0 = reduce_mean(mul(sub(Yhat0, Y), sub(Yhat0, Y)), /*axes=*/{0,1}, /*keepdims=*/false); // MSE
    float loss0 = 0.0f; for (float v : L0.value()) loss0 += v;

    // Train for a few steps
    for (int step=0; step<50; ++step) {
        auto Yhat = lin.forward(X);
        auto diff = sub(Yhat, Y);
        auto L = reduce_mean(mul(diff, diff), /*axes=*/{0,1}, /*keepdims=*/false);
        // dL/dYhat = 2*(Yhat-Y)/N  -> but reduce_mean handles scale via backward
        L.backward({1.0f});
        opt.step(lin);
        lin.zero_grad();
    }

    // Compute final loss
    auto Yhat1 = lin.forward(X);
    auto L1 = reduce_mean(mul(sub(Yhat1, Y), sub(Yhat1, Y)), /*axes=*/{0,1}, /*keepdims=*/false);
    float loss1 = 0.0f; for (float v : L1.value()) loss1 += v;

    ASSERT_TRUE(loss1 < loss0 * 0.1f); // should drop by >10x
}
