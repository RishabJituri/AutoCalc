#include "test_framework.hpp"
#include "ag/core/variables.hpp"
#include "ag/nn/loss.hpp"
#include <vector>
#include <algorithm>
#include <cmath>

using ag::Variable;
using ag::nn::cross_entropy;

static float lse3(float a,float b,float c){
    float m = std::max(a,std::max(b,c));
    return m + std::log(std::exp(a-m)+std::exp(b-m)+std::exp(c-m));
}

TEST("nn/loss/cross_entropy_forward_and_grad") {
    std::vector<float> lv = {
        1.0f, 2.0f, 3.0f,
       -1.0f, 0.5f, 0.0f
    };
    Variable logits(lv, {2,3}, /*requires_grad=*/true);
    std::vector<std::size_t> tgt = {2, 1};

    auto L = cross_entropy(logits, tgt);
    ASSERT_TRUE(L.shape().empty());

    float l0 = lse3(1.0f,2.0f,3.0f) - 3.0f;
    float l1 = lse3(-1.0f,0.5f,0.0f) - 0.5f;
    float Lref = 0.5f*(l0+l1);
    ASSERT_NEAR(L.value()[0], Lref, 1e-8f);

    L.backward({1.0f});

    auto soft = [&](int r,int c){
        float a = lv[r*3+0], b = lv[r*3+1], cc = lv[r*3+2];
        float m = std::max(a,std::max(b,cc));
        float Z = std::exp(a-m)+std::exp(b-m)+std::exp(cc-m);
        return std::exp(lv[r*3+c]-m)/Z;
    };
    for (int r=0;r<2;++r) for (int c=0;c<3;++c) {
        float g = soft(r,c) - (c==int(tgt[r]) ? 1.0f : 0.0f);
        g /= 2.0f;
        ASSERT_NEAR(logits.grad()[r*3+c], g, 1e-6f);
    }
}
