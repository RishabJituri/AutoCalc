// #include "test_framework.hpp"
// #include "ag/ops/transpose.hpp"
// #include "ag/core/variables.hpp"
// #include <vector>

// using ag::Variable;

// TEST("ops/transpose/basic_2d") {
//   // X: [2,3] = [[0,1,2],[3,4,5]]
//   std::vector<float> x = {0,1,2,3,4,5};
//   Variable X(x, {2,3}, /*requires_grad=*/true);
//   auto Y = ag::t(X); // [3,2]
//   auto sh = Y.shape();
//   ASSERT_TRUE(sh.size()==2 && sh[0]==3 && sh[1]==2);
//   // Check values at a couple indices
//   ASSERT_NEAR(Y.value()[0*2+0], 0.0f, 1e-6f); // (0,0) -> (0,0)
//   ASSERT_NEAR(Y.value()[1*2+0], 1.0f, 1e-6f); // (1,0) -> (0,1)
//   ASSERT_NEAR(Y.value()[2*2+1], 5.0f, 1e-6f); // (2,1) -> (1,2)
//   // Backward with ones
//   std::vector<float> seed(Y.value().size(), 1.0f);
//   Y.backward(seed);
//   auto gx = X.grad();
//   // Each element appears once
//   for (std::size_t i=0;i<gx.size();++i) ASSERT_NEAR(gx[i], 1.0f, 1e-6f);
// }

// TEST("ops/transpose/nd_axes") {
//   // X: [2,3,4]
//   std::vector<float> x(2*3*4);
//   for (std::size_t i=0;i<x.size();++i) x[i]=float(i);
//   Variable X(x,{2,3,4}, /*requires_grad=*/true);
//   // permute to [3,4,2]
//   auto Y = ag::transpose(X, std::vector<int>{1,2,0});
//   auto sh = Y.shape();
//   ASSERT_TRUE(sh[0]==3 && sh[1]==4 && sh[2]==2);
//   // Seed ones and ensure grads map back
//   std::vector<float> seed(Y.value().size(), 1.0f);
//   Y.backward(seed);
//   auto gx = X.grad();
//   for (std::size_t i=0;i<gx.size();++i) ASSERT_NEAR(gx[i], 1.0f, 1e-6f);
// }
