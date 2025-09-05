#include "test_framework.hpp"
#include "ag/core/variables.hpp"
#include "ag/nn/dropout.hpp"
#include <vector>
#include <cmath>

using ag::Variable;
using ag::nn::Dropout;

TEST("nn/dropout/eval_mode_is_identity") {
    std::vector<double> xv = { 0.0, 1.0, 2.0, 3.0 };
    Variable X(xv, {4}, /*requires_grad=*/true);
    Dropout d(/*p=*/0.5, /*seed=*/1234ull);
    d.eval();
    auto Y = d.forward(X);
    ASSERT_TRUE(Y.value()==xv);
    Y.backward({1,1,1,1});
    for (double g : X.grad()) ASSERT_NEAR(g, 1.0, 1e-12);
}

TEST("nn/dropout/train_mode_mask_and_scale_deterministic") {
    std::vector<double> xv = { 1.0, 2.0, 3.0, 4.0 };
    Variable X(xv, {4}, /*requires_grad=*/true);
    Dropout d(/*p=*/0.5, /*seed=*/42ull);
    d.train();
    auto Y = d.forward(X);
    for (size_t i=0;i<Y.value().size();++i) {
        ASSERT_TRUE(Y.value()[i]==0.0 || std::fabs(Y.value()[i] - 2.0*xv[i]) < 1e-12);
    }
    Y.backward({1,1,1,1});
    for (size_t i=0;i<X.grad().size();++i) {
        ASSERT_TRUE(X.grad()[i]==0.0 || std::fabs(X.grad()[i] - 2.0) < 1e-12);
    }
}
