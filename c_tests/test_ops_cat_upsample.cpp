// Tests for ag::cat and ag::upsample2d (Milestone 1)
#include "test_framework.hpp"
#include "ag/all.hpp"
#include <numeric>
#include <cmath>

using ag::Variable;

static Variable tensor(const std::vector<float>& vals, const std::vector<std::size_t>& shape, bool requires_grad = true) {
    return Variable(vals, shape, requires_grad);
}

// ===== ag::cat tests =====

TEST("cat/forward_axis0_two_matrices") {
    // A: [2,3], B: [3,3] -> cat axis 0 -> [5,3]
    auto A = tensor({1,2,3,4,5,6}, {2,3}, true);
    auto B = tensor({7,8,9,10,11,12,13,14,15}, {3,3}, true);
    auto C = ag::cat({A, B}, 0);
    auto sh = C.shape();
    ASSERT_TRUE(sh.size() == 2);
    ASSERT_TRUE(sh[0] == 5);
    ASSERT_TRUE(sh[1] == 3);
    auto d = C.value();
    float expect[] = {1,2,3,4,5,6,7,8,9,10,11,12,13,14,15};
    for (int i = 0; i < 15; i++) ASSERT_NEAR(d[i], expect[i], 1e-6f);
}

TEST("cat/forward_axis1") {
    // A: [2,2], B: [2,3] -> cat axis 1 -> [2,5]
    auto A = tensor({1,2,3,4}, {2,2}, true);
    auto B = tensor({5,6,7,8,9,10}, {2,3}, true);
    auto C = ag::cat({A, B}, 1);
    auto sh = C.shape();
    ASSERT_TRUE(sh.size() == 2);
    ASSERT_TRUE(sh[0] == 2);
    ASSERT_TRUE(sh[1] == 5);
    // Row-major: row0=[1,2,5,6,7], row1=[3,4,8,9,10]
    auto d = C.value();
    float expect[] = {1,2,5,6,7,3,4,8,9,10};
    for (int i = 0; i < 10; i++) ASSERT_NEAR(d[i], expect[i], 1e-6f);
}

TEST("cat/backward_axis0") {
    auto A = tensor({1,2,3,4}, {2,2}, true);
    auto B = tensor({5,6,7,8,9,10}, {3,2}, true);
    auto C = ag::cat({A, B}, 0);  // [5,2]
    auto S = ag::reduce_sum(C);
    S.backward();
    auto gA = A.grad();
    auto gB = B.grad();
    for (std::size_t i = 0; i < 4; i++) ASSERT_NEAR(gA[i], 1.0f, 1e-6f);
    for (std::size_t i = 0; i < 6; i++) ASSERT_NEAR(gB[i], 1.0f, 1e-6f);
}

TEST("cat/backward_axis1_weighted") {
    auto A = tensor({1,2,3,4}, {2,2}, true);
    auto B = tensor({5,6,7,8}, {2,2}, true);
    auto C = ag::cat({A, B}, 1);  // [2,4]
    auto two = tensor({2,2,2,2,2,2,2,2}, {2,4}, false);
    auto D = ag::mul(C, two);
    auto S = ag::reduce_sum(D);
    S.backward();
    auto gA = A.grad();
    auto gB = B.grad();
    for (std::size_t i = 0; i < 4; i++) ASSERT_NEAR(gA[i], 2.0f, 1e-6f);
    for (std::size_t i = 0; i < 4; i++) ASSERT_NEAR(gB[i], 2.0f, 1e-6f);
}

TEST("cat/three_inputs") {
    auto A = tensor({1,2}, {1,2}, true);
    auto B = tensor({3,4}, {1,2}, true);
    auto C = tensor({5,6}, {1,2}, true);
    auto D = ag::cat({A, B, C}, 0);  // [3,2]
    auto sh = D.shape();
    ASSERT_TRUE(sh[0] == 3);
    ASSERT_TRUE(sh[1] == 2);
    auto d = D.value();
    float expect[] = {1,2,3,4,5,6};
    for (int i = 0; i < 6; i++) ASSERT_NEAR(d[i], expect[i], 1e-6f);
}

TEST("cat/negative_axis") {
    auto A = tensor({1,2,3,4}, {2,2}, true);
    auto B = tensor({5,6,7,8}, {2,2}, true);
    auto C = ag::cat({A, B}, -1);  // [2,4]
    auto sh = C.shape();
    ASSERT_TRUE(sh[0] == 2);
    ASSERT_TRUE(sh[1] == 4);
}

// ===== ag::upsample2d tests =====

TEST("upsample2d/forward_2x2") {
    auto X = tensor({1,2,3,4}, {1,1,2,2}, true);
    auto Y = ag::upsample2d(X, 2, 2);
    auto sh = Y.shape();
    ASSERT_TRUE(sh.size() == 4);
    ASSERT_TRUE(sh[0] == 1);
    ASSERT_TRUE(sh[1] == 1);
    ASSERT_TRUE(sh[2] == 4);
    ASSERT_TRUE(sh[3] == 4);
    auto d = Y.value();
    // nearest neighbor: each pixel replicated to 2x2 block
    float expect[] = {1,1,2,2, 1,1,2,2, 3,3,4,4, 3,3,4,4};
    for (int i = 0; i < 16; i++) ASSERT_NEAR(d[i], expect[i], 1e-6f);
}

TEST("upsample2d/forward_scale_3x3") {
    auto X = tensor({10, 20}, {1,1,1,2}, true);
    auto Y = ag::upsample2d(X, 3, 3);
    auto sh = Y.shape();
    ASSERT_TRUE(sh[2] == 3);
    ASSERT_TRUE(sh[3] == 6);
    auto d = Y.value();
    for (int r = 0; r < 3; r++) {
        for (int c = 0; c < 3; c++) ASSERT_NEAR(d[r*6+c], 10.0f, 1e-6f);
        for (int c = 3; c < 6; c++) ASSERT_NEAR(d[r*6+c], 20.0f, 1e-6f);
    }
}

TEST("upsample2d/backward_sums_into_source") {
    auto X = tensor({1,2,3,4}, {1,1,2,2}, true);
    auto Y = ag::upsample2d(X, 2, 2);  // [1,1,4,4]
    auto S = ag::reduce_sum(Y);
    S.backward();
    auto gX = X.grad();
    for (std::size_t i = 0; i < 4; i++) ASSERT_NEAR(gX[i], 4.0f, 1e-6f);
}

TEST("upsample2d/backward_weighted") {
    auto X = tensor({1,2}, {1,1,1,2}, true);
    auto Y = ag::upsample2d(X, 2, 2);  // [1,1,2,4]
    std::vector<float> w(8);
    for (int i = 0; i < 8; i++) w[i] = static_cast<float>(i + 1);
    auto W = tensor(w, {1,1,2,4}, false);
    auto Z = ag::mul(Y, W);
    auto S = ag::reduce_sum(Z);
    S.backward();
    auto gX = X.grad();
    // grad for pixel(0,0) = W[0,0]+W[0,1]+W[1,0]+W[1,1] = 1+2+5+6 = 14
    // grad for pixel(0,1) = W[0,2]+W[0,3]+W[1,2]+W[1,3] = 3+4+7+8 = 22
    ASSERT_NEAR(gX[0], 14.0f, 1e-5f);
    ASSERT_NEAR(gX[1], 22.0f, 1e-5f);
}

TEST("upsample2d/multi_batch_channel") {
    std::vector<float> buf(6);
    for (int i = 0; i < 6; i++) buf[i] = static_cast<float>(i + 1);
    auto X = tensor(buf, {2,3,1,1}, true);
    auto Y = ag::upsample2d(X, 2, 2);
    auto sh = Y.shape();
    ASSERT_TRUE(sh[0] == 2);
    ASSERT_TRUE(sh[1] == 3);
    ASSERT_TRUE(sh[2] == 2);
    ASSERT_TRUE(sh[3] == 2);
    auto d = Y.value();
    for (int bc = 0; bc < 6; bc++) {
        float v = static_cast<float>(bc + 1);
        for (int p = 0; p < 4; p++) {
            ASSERT_NEAR(d[bc*4 + p], v, 1e-6f);
        }
    }
}
