// // filepath: tests/test_nn_module_sgd_resnet_linear.cpp
// #include "test_framework.hpp"
// #include "ag/core/variables.hpp"
// #include "ag/ops/activations.hpp"
// #include "ag/nn/layers/linear.hpp"
// #include "ag/nn/loss.hpp"
// #include "ag/nn/optim/sgd.hpp"
// #include <random>
// #include <vector>
// #include <cstddef>
// #include <cmath>

// using ag::Variable;
// using ag::nn::Linear;
// using ag::nn::SGD;

// // Residual block: hidden-size -> hidden-size with skip connection
// struct ResBlock : ag::nn::Module {
//   Linear a{0,0};
//   Linear b{0,0};
//   ResBlock() = default;
//   ResBlock(std::size_t dim, float init_scale=0.05f, unsigned long long seed=0xDEADBEEFull)
//     : a(dim, dim, /*bias=*/true, init_scale, seed),
//       b(dim, dim, /*bias=*/true, init_scale, seed*2ULL) {
//     register_module("a", a);
//     register_module("b", b);
//   }
//   Variable forward(const Variable& x) override {
//     auto h = ag::relu(a.forward(x));
//     h = b.forward(h);
//     // elementwise add (skip)
//     return ag::relu(ag::add(h, x));
//   }
// protected:
//   std::vector<ag::Variable*> _parameters() override { return {}; }
// };

// // Small ResNet-style classifier: 2 -> hidden -> (N blocks) -> 2
// struct ResNetSmall : ag::nn::Module {
//   Linear in{2, 32, /*bias=*/true, /*init_scale=*/0.05f, /*seed=*/1234ull};
//   std::vector<std::shared_ptr<ResBlock>> blocks;
//   Linear out{32, 2, /*bias=*/true, /*init_scale=*/0.05f, /*seed=*/5678ull};

//   ResNetSmall(std::size_t nblocks=3) {
//     register_module("in", in);
//     for (std::size_t i = 0; i < nblocks; ++i) {
//       blocks.emplace_back(std::make_shared<ResBlock>(32, 0.05f, 1234ull + i));
//       // register child module (owning)
//       register_module(std::string("block") + std::to_string(i), blocks.back());
//     }
//     register_module("out", out);
//   }

//   Variable forward(const Variable& x) override {
//     auto h = ag::relu(in.forward(x));
//     for (auto& b : blocks) h = b->forward(h);
//     return out.forward(h);
//   }
// protected:
//   std::vector<ag::Variable*> _parameters() override { return {}; }
// };

// static void make_linear_plane_dataset(std::size_t B,
//                                       std::vector<float>& X,
//                                       std::vector<std::size_t>& y) {
//   std::mt19937 rng(42);
//   std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
//   X.resize(B * 2);
//   y.resize(B);
//   // True separating plane: 0.6*x0 - 0.4*x1 + 0.05
//   for (std::size_t i = 0; i < B; ++i) {
//     float x0 = dist(rng);
//     float x1 = dist(rng);
//     X[2*i + 0] = x0;
//     X[2*i + 1] = x1;
//     float s = 0.6f * x0 - 0.4f * x1 + 0.05f;
//     y[i] = (s >= 0.0f) ? 1u : 0u;
//   }
// }

// TEST("nn/module_sgd_resnet_linear_converges") {
//   const std::size_t B = 512;
//   std::vector<float> xv; std::vector<std::size_t> yv;
//   make_linear_plane_dataset(B, xv, yv);

//   ResNetSmall net(/*nblocks=*/3);
//   SGD opt(/*lr=*/0.2f, /*momentum=*/0.9f, /*nesterov=*/true, /*weight_decay=*/0.0f);

//   Variable X(xv, {B, 2}, /*requires_grad=*/true);

//   // Train
//   for (int step = 0; step < 400; ++step) {
//     auto logits = net.forward(X);
//     auto loss = ag::nn::cross_entropy(logits, yv);
//     loss.backward();
//     opt.step(net);
//   }

//   // Evaluate
//   auto logits = net.forward(X);
//   const auto& z = logits.value();
//   std::size_t correct = 0;
//   for (std::size_t i = 0; i < B; ++i) {
//     float z0 = z[2*i + 0];
//     float z1 = z[2*i + 1];
//     std::size_t pred = (z1 > z0) ? 1u : 0u;
//     if (pred == yv[i]) ++correct;
//   }
//   float acc = float(correct) / float(B);
//   // Record whether it converged to high accuracy
//   ASSERT_GE(acc, 0.98f);
// }
