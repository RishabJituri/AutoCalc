#include "test_framework.hpp"
#include "ag/core/variables.hpp"
#include "ag/nn/loss.hpp"
#include <vector>
#include <algorithm>
#include <cmath>

using ag::Variable;
using ag::nn::cross_entropy;

static double lse3(double a,double b,double c){
    double m = std::max(a,std::max(b,c));
    return m + std::log(std::exp(a-m)+std::exp(b-m)+std::exp(c-m));
}

TEST("nn/loss/cross_entropy_forward_and_grad") {
    std::vector<double> lv = {
        1.0, 2.0, 3.0,
       -1.0, 0.5, 0.0
    };
    Variable logits(lv, {2,3}, /*requires_grad=*/true);
    std::vector<std::size_t> tgt = {2, 1};

    auto L = cross_entropy(logits, tgt);
    ASSERT_TRUE(L.shape().empty());

    double l0 = lse3(1.0,2.0,3.0) - 3.0;
    double l1 = lse3(-1.0,0.5,0.0) - 0.5;
    double Lref = 0.5*(l0+l1);
    ASSERT_NEAR(L.value()[0], Lref, 1e-8);

    L.backward({1.0});

    auto soft = [&](int r,int c){
        double a = lv[r*3+0], b = lv[r*3+1], cc = lv[r*3+2];
        double m = std::max(a,std::max(b,cc));
        double Z = std::exp(a-m)+std::exp(b-m)+std::exp(cc-m);
        return std::exp(lv[r*3+c]-m)/Z;
    };
    for (int r=0;r<2;++r) for (int c=0;c<3;++c) {
        double g = soft(r,c) - (c==int(tgt[r]) ? 1.0 : 0.0);
        g /= 2.0;
        ASSERT_NEAR(logits.grad()[r*3+c], g, 1e-6);
    }
}
