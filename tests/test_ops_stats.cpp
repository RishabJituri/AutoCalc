// #include "test_framework.hpp"
// #include "ag/core/variables.hpp"
// #include "ag/ops/stats.hpp"
// #include "ag/ops/activations.hpp"
// #include "ag/ops/reduce.hpp"
// #include <vector>
// #include <algorithm>
// #include <cmath>

// using ag::Variable;
// using ag::reduce_max;
// using ag::logsumexp;
// using ag::softmax;
// using ag::argmax_lastdim;

// TEST("ops/stats/reduce_max_keepdims_false_and_backward_ties") {
//     std::vector<float> xv = {
//         1.0f, 5.0f, 5.0f,
//         -1.0f, 0.0f, 2.0f
//     };
//     Variable X(xv, {2,3}, /*requires_grad=*/true);
//     auto Y = reduce_max(X, /*axes=*/{1}, /*keepdims=*/false);
//     ASSERT_TRUE(Y.shape().size()==1 && Y.shape()[0]==2);
//     ASSERT_NEAR(Y.value()[0], 5.0f, 1e-6f);
//     ASSERT_NEAR(Y.value()[1], 2.0f, 1e-6f);

//     Y.backward({1.0f,1.0f});
//     ASSERT_NEAR(X.grad()[1], 0.5f, 1e-6f);
//     ASSERT_NEAR(X.grad()[2], 0.5f, 1e-6f);
//     ASSERT_NEAR(X.grad()[5], 1.0f, 1e-6f);
// }

// TEST("ops/stats/logsumexp_matches_reference") {
//     std::vector<float> xv = { -5.0f, 0.0f, 5.0f, 10.0f };
//     Variable X(xv, {2,2}, /*requires_grad=*/false);
//     auto LSE = logsumexp(X, /*axes=*/{1}, /*keepdims=*/false);
//     for (int r=0;r<2;++r) {
//         float a = X.value()[r*2+0], b = X.value()[r*2+1];
//         float m = std::max(a,b);
//         float exp_sum = std::exp(a-m) + std::exp(b-m);
//         float ref = m + std::log(exp_sum);
//         ASSERT_NEAR(LSE.value()[r], ref, 1e-6f);
//     }
// }

// TEST("ops/stats/softmax_rows_sum_to_one") {
//     std::vector<float> xv = { 0.1f, -0.2f, 0.3f,
//                                2.0f,  0.0f, -1.0f };
//     Variable X(xv, {2,3}, false);
//     auto Y = softmax(X, /*axis=*/1);
//     for (int r=0;r<2;++r) {
//         float s=0.0f; for (int c=0;c<3;++c) s += Y.value()[r*3 + c];
//         ASSERT_NEAR(s, 1.0f, 1e-6f);
//     }
// }

// TEST("ops/stats/argmax_lastdim_returns_indices") {
//     std::vector<float> xv = { 0.2f, 0.5f, 0.3f,
//                                1.0f, -2.0f, 3.0f };
//     Variable X(xv, {2,3}, false);
//     auto idx = argmax_lastdim(X);
//     ASSERT_TRUE(idx.size()==2);
//     ASSERT_TRUE(idx[0]==1);
//     ASSERT_TRUE(idx[1]==2);
// }
