// tests/test_nn_conv2d.cpp (fixed names)
#include "test_framework.hpp"
#include "ag/nn/layers/conv2d.hpp"
#include "ag/core/variables.hpp"
#include <vector>
#include <cmath>

using ag::Variable;
using ag::nn::Conv2d;

static std::vector<double> zeros(std::size_t n) { return std::vector<double>(n, 0.0); }
static std::vector<double> ones(std::size_t n)  { return std::vector<double>(n, 1.0); }

TEST("nn/conv2d/identity_1x1_no_bias") {
    const std::size_t B=1, C=1, H=2, Wimg=3;
    Conv2d conv(/*Cin*/1, /*Cout*/1, /*kernel*/{1,1}, /*stride*/{1,1}, /*pad*/{0,0}, /*dil*/{1,1}, /*bias*/false);

    // Set weight tensor to 1.0
    auto params = conv.parameters();
    Variable& Wparam = *params[0];
    for (auto& v : Wparam.n->value) v = 1.0;

    // x arbitrary
    std::vector<double> xv = {1,2,3,4,5,6};
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
    for (std::size_t oc=0; oc<Cout; ++oc) b.n->value[oc] = double(oc+1);

    Variable x(zeros(B*Cin*H*Wimg), {B,Cin,H,Wimg}, /*requires_grad=*/true);
    auto y = conv.forward(x);

    // Every position in channel oc should equal b[oc]
    for (std::size_t bidx=0; bidx<B; ++bidx) {
      for (std::size_t oc=0; oc<Cout; ++oc) {
        for (std::size_t i=0;i<H*Wimg;++i) {
          const std::size_t off = ((bidx*Cout + oc)*H + (i/ Wimg))*Wimg + (i%Wimg);
          ASSERT_NEAR(y.value()[off], double(oc+1), 1e-12);
        }
      }
    }

    // Backward with seed=1: dW=0, db = B*H*Wimg
    y.backward(ones(B*Cout*H*Wimg));
    for (double gw : Wparam.grad()) ASSERT_NEAR(gw, 0.0, 1e-12);
    for (std::size_t oc=0; oc<Cout; ++oc) {
      ASSERT_NEAR(b.grad()[oc], double(B*H*Wimg), 1e-9);
    }
}

TEST("nn/conv2d/numeric_grad_wrt_x_sum") {
    const std::size_t B=1, Cin=1, H=3, Wimg=3, Cout=1;
    Conv2d conv(Cin, Cout, {3,3}, {1,1}, {0,0}, {1,1}, /*bias=*/true, /*init_scale=*/0.1, /*seed=*/42ull);

    // Prepare x
    std::vector<double> xv(H*Wimg);
    for (std::size_t i=0;i<xv.size();++i) xv[i] = double(i+1)*0.1;
    Variable x(xv, {B,Cin,H,Wimg}, /*requires_grad=*/true);

    // y = conv(x); L = sum(y)
    auto y = conv.forward(x);
    y.backward(ones(y.value().size()));
    auto gx = x.grad();

    // finite-diff a couple entries
    const double eps=1e-6;
    for (std::size_t check=0; check<std::min<std::size_t>(4, xv.size()); ++check) {
        auto xp = xv; xp[check]+=eps;
        auto xm = xv; xm[check]-=eps;
        Variable Xp(xp, {B,Cin,H,Wimg}, /*requires_grad=*/false);
        Variable Xm(xm, {B,Cin,H,Wimg}, /*requires_grad=*/false);
        auto yp = conv.forward(Xp);
        auto ym = conv.forward(Xm);
        double fp=0.0,fm=0.0;
        for (double v : yp.value()) fp+=v;
        for (double v : ym.value()) fm+=v;
        double gnum = (fp-fm)/(2*eps);
        ASSERT_NEAR(gx[check], gnum, 1e-4);
    }
}
