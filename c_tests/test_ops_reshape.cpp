// #include "test_framework.hpp"
// #include "ag/core/variables.hpp"
// #include "ag/ops/reshape.hpp"
// #include <vector>

// using ag::Variable;
// using ag::flatten;

// TEST("ops/reshape/flatten_forward_and_backward") {
//     std::vector<float> xv = {1,2,3,4,5,6};
//     Variable X(xv, {2,3}, /*requires_grad=*/true);
//     auto Y = flatten(X, /*start_dim=*/1);

//     ASSERT_TRUE(Y.shape().size()==2 && Y.shape()[0]==2 && Y.shape()[1]==3);
//     for (size_t i=0;i<xv.size();++i) ASSERT_NEAR(Y.value()[i], xv[i], 1e-12);

//     std::vector<float> seed(Y.value().size(), 1.0f);
//     Y.backward(seed);
//     for (float g : X.grad()) ASSERT_NEAR(g, 1.0f, 1e-12f);
// }
