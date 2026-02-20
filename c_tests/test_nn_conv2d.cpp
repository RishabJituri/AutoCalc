// tests/test_nn_conv2d.cpp (fixed names)
#include "test_framework.hpp"
#include "ag/nn/layers/conv2d.hpp"
#include "ag/core/variables.hpp"
#include <vector>
#include <cmath>

using ag::Variable;
using ag::nn::Conv2d;

static std::vector<float> zeros(std::size_t n) { return std::vector<float>(n, 0.0f); }
static std::vector<float> ones(std::size_t n)  { return std::vector<float>(n, 1.0f); }

TEST("nn/conv2d/identity_1x1_no_bias") {
    const std::size_t B=1, C=1, H=2, Wimg=3;
    Conv2d conv(/*Cin*/1, /*Cout*/1, /*kernel*/{1,1}, /*stride*/{1,1}, /*pad*/{0,0}, /*dil*/{1,1}, /*bias*/false);

    // Set weight tensor to 1.0
    auto params = conv.parameters();
    Variable& Wparam = *params[0];
    for (auto& v : Wparam.n->value) v = 1.0;

    // x arbitrary
    std::vector<float> xv = {1,2,3,4,5,6};
    Variable x(xv, {B,C,H,Wimg}, /*requires_grad=*/true);

    auto y = conv.forward(x);
    ASSERT_TRUE(y.shape()[0]==B && y.shape()[1]==1 && y.shape()[2]==H && y.shape()[3]==Wimg);
    for (std::size_t i=0;i<xv.size();++i) ASSERT_NEAR(y.value()[i], xv[i], 1e-12);
}

TEST("nn/conv2d/bias_fill_when_x_zero") {
    const std::size_t B=2, Cin=2, H=3, Wimg=3, Cout=3;
    Conv2d conv(Cin, Cout, {3,3}, {1,1}, {1,1}, {1,1}, /*bias=*/true);

    auto params = conv.parameters();
    Variable& Wparam = *params[0];
    Variable& b = *params[1];
    // Zero weights; bias set to increasing values
    for (auto& v : Wparam.n->value) v = 0.0;
    for (std::size_t oc=0; oc<Cout; ++oc) b.n->value[oc] = float(oc+1);

    Variable x(zeros(B*Cin*H*Wimg), {B,Cin,H,Wimg}, /*requires_grad=*/true);
    auto y = conv.forward(x);

    // Every position in channel oc should equal b[oc]
    for (std::size_t bidx=0; bidx<B; ++bidx) {
      for (std::size_t oc=0; oc<Cout; ++oc) {
        for (std::size_t i=0;i<H*Wimg;++i) {
          const std::size_t off = ((bidx*Cout + oc)*H + (i/ Wimg))*Wimg + (i%Wimg);
          ASSERT_NEAR(y.value()[off], float(oc+1), 1e-6f);
        }
      }
    }

    // Backward with seed=1: dW=0, db = B*H*Wimg
    y.backward(ones(B*Cout*H*Wimg));
    for (float gw : Wparam.grad()) ASSERT_NEAR(gw, 0.0, 1e-12);
    for (std::size_t oc=0; oc<Cout; ++oc) {
      ASSERT_NEAR(b.grad()[oc], float(B*H*Wimg), 1e-6f);
    }
}

TEST("nn/conv2d/numeric_grad_wrt_x_sum") {
    const std::size_t B=1, Cin=1, H=3, Wimg=3, Cout=1;
    Conv2d conv(Cin, Cout, {3,3}, {1,1}, {0,0}, {1,1}, /*bias=*/true, /*init_scale=*/0.1, /*seed=*/42ull);

    // Prepare x
    std::vector<float> xv(H*Wimg);
    for (std::size_t i=0;i<xv.size();++i) xv[i] = float(i+1)*0.1f;
    Variable x(xv, {B,Cin,H,Wimg}, /*requires_grad=*/true);

    // y = conv(x); L = sum(y)
    auto y = conv.forward(x);
    y.backward(ones(y.value().size()));
    auto gx = x.grad();

    // finite-diff a couple entries
    const float eps=1e-4f;
    for (std::size_t check=0; check<std::min<std::size_t>(4, xv.size()); ++check) {
        auto xp = xv; xp[check]+=eps;
        auto xm = xv; xm[check]-=eps;
        Variable Xp(xp, {B,Cin,H,Wimg}, /*requires_grad=*/false);
        Variable Xm(xm, {B,Cin,H,Wimg}, /*requires_grad=*/false);
        auto yp = conv.forward(Xp);
        auto ym = conv.forward(Xm);
        float fp=0.0f,fm=0.0f;
        for (float v : yp.value()) fp+=v;
        for (float v : ym.value()) fm+=v;
        float gnum = (fp-fm)/(2*eps);
        
        ASSERT_NEAR(gx[check], gnum, 1e-3f);
    }
}

TEST("nn/conv2d/numeric_grad_wrt_weight_and_bias") {
    // Test analytic vs numeric gradients for weights and bias
    const std::size_t B=1, Cin=1, H=4, Wimg=4, Cout=1;
    Conv2d conv(Cin, Cout, {3,3}, {1,1}, {1,1}, {1,1}, /*bias=*/true, /*init_scale=*/0.1, /*seed=*/123ull);

    // Prepare x
    std::vector<float> xv(B*Cin*H*Wimg);
    for (std::size_t i=0;i<xv.size();++i) xv[i] = float(i+1)*0.05f;
    Variable x(xv, {B,Cin,H,Wimg}, /*requires_grad=*/false);

    // Forward
    auto y = conv.forward(x);
    y.backward(ones(y.value().size()));

    // Numeric grad for weights
    auto params = conv.parameters();
    Variable& Wparam = *params[0];
    Variable& bparam = *params[1];
    auto gw = Wparam.grad();
    auto gb = bparam.grad();
    const float eps=1e-4f;
    for (std::size_t i=0; i<Wparam.value().size(); ++i) {
        auto orig = Wparam.n->value[i];
        Wparam.n->value[i] = orig + eps;
        auto yp = conv.forward(x);
        float fp=0.0f; for (float v : yp.value()) fp+=v;
        Wparam.n->value[i] = orig - eps;
        auto ym = conv.forward(x);
        float fm=0.0f; for (float v : ym.value()) fm+=v;
        float gnum = (fp-fm)/(2*eps);
        Wparam.n->value[i] = orig;
        ASSERT_NEAR(gw[i], gnum, 1e-3f);
    }
    // Numeric grad for bias
    for (std::size_t i=0; i<bparam.value().size(); ++i) {
        auto orig = bparam.n->value[i];
        bparam.n->value[i] = orig + eps;
        auto yp = conv.forward(x);
        float fp=0.0f; for (float v : yp.value()) fp+=v;
        bparam.n->value[i] = orig - eps;
        auto ym = conv.forward(x);
        float fm=0.0f; for (float v : ym.value()) fm+=v;
        float gnum = (fp-fm)/(2*eps);
        bparam.n->value[i] = orig;
        ASSERT_NEAR(gb[i], gnum, 1e-3f);
    }
}

TEST("nn/conv2d/backward_wrt_x_multibatch_multichannel") {
    // Test backward wrt input for multi-batch, multi-channel
    const std::size_t B=2, Cin=2, H=3, Wimg=3, Cout=2;
    Conv2d conv(Cin, Cout, {2,2}, {1,1}, {0,0}, {1,1}, /*bias=*/true, /*init_scale=*/0.2, /*seed=*/321ull);

    std::vector<float> xv(B*Cin*H*Wimg);
    for (std::size_t i=0;i<xv.size();++i) xv[i] = float(i+1)*0.01f;
    Variable x(xv, {B,Cin,H,Wimg}, /*requires_grad=*/true);

    auto y = conv.forward(x);
    y.backward(ones(y.value().size()));
    auto gx = x.grad();

    // Numeric gradient check for a few entries
    const float eps=1e-4f;
    for (std::size_t check=0; check<std::min<std::size_t>(5, xv.size()); ++check) {
        auto xp = xv; xp[check]+=eps;
        auto xm = xv; xm[check]-=eps;
        Variable Xp(xp, {B,Cin,H,Wimg}, /*requires_grad=*/false);
        Variable Xm(xm, {B,Cin,H,Wimg}, /*requires_grad=*/false);
        auto yp = conv.forward(Xp);
        auto ym = conv.forward(Xm);
        float fp=0.0f,fm=0.0f;
        for (float v : yp.value()) fp+=v;
        for (float v : ym.value()) fm+=v;
        float gnum = (fp-fm)/(2*eps);
        ASSERT_NEAR(gx[check], gnum, 1e-3f);
    }
}

TEST("nn/conv2d/backward_wrt_x_stride_and_padding") {
    // Test backward wrt input with stride and padding
    const std::size_t B=1, Cin=1, H=5, Wimg=5, Cout=1;
    Conv2d conv(Cin, Cout, {3,3}, {2,2}, {1,1}, {1,1}, /*bias=*/false, /*init_scale=*/0.15, /*seed=*/99ull);

    std::vector<float> xv(B*Cin*H*Wimg);
    for (std::size_t i=0;i<xv.size();++i) xv[i] = float(i+1)*0.02f;
    Variable x(xv, {B,Cin,H,Wimg}, /*requires_grad=*/true);

    auto y = conv.forward(x);
    y.backward(ones(y.value().size()));
    auto gx = x.grad();

    // Numeric gradient check for a few entries
    const float eps=1e-4f;
    for (std::size_t check=0; check<std::min<std::size_t>(4, xv.size()); ++check) {
        auto xp = xv; xp[check]+=eps;
        auto xm = xv; xm[check]-=eps;
        Variable Xp(xp, {B,Cin,H,Wimg}, /*requires_grad=*/false);
        Variable Xm(xm, {B,Cin,H,Wimg}, /*requires_grad=*/false);
        auto yp = conv.forward(Xp);
        auto ym = conv.forward(Xm);
        float fp=0.0f,fm=0.0f;
        for (float v : yp.value()) fp+=v;
        for (float v : ym.value()) fm+=v;
        float gnum = (fp-fm)/(2*eps);
        ASSERT_NEAR(gx[check], gnum, 1e-3f);
    }
}

TEST("nn/conv2d/backward_wrt_x_dilation") {
    // Test backward wrt input with dilation
    const std::size_t B=1, Cin=1, H=6, Wimg=6, Cout=1;
    Conv2d conv(Cin, Cout, {3,3}, {1,1}, {0,0}, {2,2}, /*bias=*/false, /*init_scale=*/0.12, /*seed=*/77ull);

    std::vector<float> xv(B*Cin*H*Wimg);
    for (std::size_t i=0;i<xv.size();++i) xv[i] = float(i+1)*0.03f;
    Variable x(xv, {B,Cin,H,Wimg}, /*requires_grad=*/true);

    auto y = conv.forward(x);
    y.backward(ones(y.value().size()));
    auto gx = x.grad();

    // Numeric gradient check for a few entries
    const float eps=1e-4f;
    for (std::size_t check=0; check<std::min<std::size_t>(4, xv.size()); ++check) {
        auto xp = xv; xp[check]+=eps;
        auto xm = xv; xm[check]-=eps;
        Variable Xp(xp, {B,Cin,H,Wimg}, /*requires_grad=*/false);
        Variable Xm(xm, {B,Cin,H,Wimg}, /*requires_grad=*/false);
        auto yp = conv.forward(Xp);
        auto ym = conv.forward(Xm);
        float fp=0.0f,fm=0.0f;
        for (float v : yp.value()) fp+=v;
        for (float v : ym.value()) fm+=v;
        float gnum = (fp-fm)/(2*eps);
        ASSERT_NEAR(gx[check], gnum, 1e-3f);
    }
}
