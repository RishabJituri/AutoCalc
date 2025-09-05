#include "test_framework.hpp"
#include "ag/core/variables.hpp"
#include "ag/nn/layers/pooling.hpp"
#include <vector>

using ag::Variable;
using ag::nn::MaxPool2d;
using ag::nn::AvgPool2d;

static std::vector<double> ones(std::size_t n){ return std::vector<double>(n,1.0); }

TEST("nn/pooling/maxpool2d_forward_basic") {
    std::vector<double> xv = {
         1, 2, 3,
         4, 5, 6,
         7, 8, 9
    };
    Variable X(xv, {1,1,3,3}, /*requires_grad=*/true);
    MaxPool2d pool(2,2,1,1,0,0);
    auto Y = pool.forward(X);
    ASSERT_TRUE(Y.shape()==std::vector<std::size_t>({1,1,2,2}));
    ASSERT_NEAR(Y.value()[0], 5.0, 1e-12);
    ASSERT_NEAR(Y.value()[1], 6.0, 1e-12);
    ASSERT_NEAR(Y.value()[2], 8.0, 1e-12);
    ASSERT_NEAR(Y.value()[3], 9.0, 1e-12);
}

TEST("nn/pooling/maxpool2d_backward_tie_splitting") {
    std::vector<double> xv = { 1,2, 2,1 };
    Variable X(xv, {1,1,2,2}, /*requires_grad=*/true);
    MaxPool2d pool(2,2,1,1,0,0);
    auto Y = pool.forward(X);
    Y.backward(ones(Y.value().size()));
    ASSERT_NEAR(X.grad()[1], 0.5, 1e-12);
    ASSERT_NEAR(X.grad()[2], 0.5, 1e-12);
}

TEST("nn/pooling/avgpool2d_forward_backward") {
    std::vector<double> xv = { 1,2,3,4 };
    Variable X(xv, {1,1,2,2}, /*requires_grad=*/true);
    AvgPool2d pool(2,2,2,2,0,0);
    auto Y = pool.forward(X);
    ASSERT_TRUE(Y.shape()==std::vector<std::size_t>({1,1,1,1}));
    ASSERT_NEAR(Y.value()[0], (1+2+3+4)/4.0, 1e-12);
    Y.backward({1.0});
    for (double g : X.grad()) ASSERT_NEAR(g, 0.25, 1e-12);
}
