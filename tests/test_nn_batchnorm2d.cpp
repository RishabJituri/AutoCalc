#include "test_framework.hpp"
#include "ag/core/variables.hpp"
#include "ag/nn/layers/normalization.hpp"
#include <vector>
#include <cmath>

using ag::Variable;
using ag::nn::BatchNorm2d;

static std::vector<double> ones(std::size_t n){ return std::vector<double>(n,1.0); }

TEST("nn/batchnorm2d/forward_training_default_affine") {
    std::vector<double> xv = { 1.0,  2.0,  3.0,
                               -1.0, 0.0,  1.0 };
    Variable X(xv, {1,2,1,3}, /*requires_grad=*/true);
    BatchNorm2d bn(/*C=*/2, /*eps=*/1e-5, /*momentum=*/0.1);
    bn.train();
    auto Y = bn.forward(X);

    double m0 = (1+2+3)/3.0, v0 = ((1-m0)*(1-m0)+(2-m0)*(2-m0)+(3-m0)*(3-m0))/3.0;
    double m1 = (-1+0+1)/3.0, v1 = ((-1-m1)*(-1-m1)+m1*m1+(1-m1)*(1-m1))/3.0;
    double is0 = 1.0/std::sqrt(v0+1e-5), is1 = 1.0/std::sqrt(v1+1e-5);

    ASSERT_NEAR(Y.value()[0], (1.0 - m0)*is0, 1e-5);
    ASSERT_NEAR(Y.value()[1], (2.0 - m0)*is0, 1e-5);
    ASSERT_NEAR(Y.value()[2], (3.0 - m0)*is0, 1e-5);
    ASSERT_NEAR(Y.value()[3], (-1.0 - m1)*is1, 1e-5);
    ASSERT_NEAR(Y.value()[4], (0.0  - m1)*is1, 1e-5);
    ASSERT_NEAR(Y.value()[5], (1.0  - m1)*is1, 1e-5);
}

TEST("nn/batchnorm2d/numeric_input_grad_sum") {
    const std::size_t B=1, C=2, H=1, W=3;
    std::vector<double> xv = { 1.0,2.0,3.0, -1.0,0.0,1.0 };
    Variable X(xv, {B,C,H,W}, /*requires_grad=*/true);
    BatchNorm2d bn(C, 1e-5, 0.1);
    bn.train();
    auto Y = bn.forward(X);

    Y.backward(ones(Y.value().size()));
    auto gx = X.grad();

    const double eps=1e-6;
    for (size_t i=0;i<xv.size();++i) {
        auto xp = xv; xp[i]+=eps;
        auto xm = xv; xm[i]-=eps;
        Variable Xp(xp, {B,C,H,W}, false);
        Variable Xm(xm, {B,C,H,W}, false);
        auto Yp = bn.forward(Xp);
        auto Ym = bn.forward(Xm);
        double Lp=0,Lm=0; for(double v:Yp.value()) Lp+=v; for(double v:Ym.value()) Lm+=v;
        double gnum = (Lp-Lm)/(2*eps);
        ASSERT_NEAR(gx[i], gnum, 1e-3);
    }
}
