// // filepath: tests/test_nn_module_sgd_spiral_multiclass.cpp
// #include "test_framework.hpp"
// #include "ag/core/variables.hpp"
// #include "ag/nn/layers/linear.hpp"
// #include "ag/nn/loss.hpp"
// #include "ag/nn/optim/sgd.hpp"
// #include "ag/ops/activations.hpp"
// #include <vector>
// #include <random>
// #include <cstddef>
// #include <cmath>

// using ag::Variable;
// using ag::nn::Linear;
// using ag::nn::SGD;

// // Two-layer MLP for multi-class spiral classification: 2 -> H -> K
// struct SpiralNet : ag::nn::Module {
//   // Increased hidden width and slightly larger init scale to improve capacity
//   Linear l1{2, 128, /*bias=*/true, /*init_scale=*/0.5f, /*seed=*/12345ull};
//   Linear l2{128, 3, /*bias=*/true, /*init_scale=*/0.5f, /*seed=*/56789ull};
//   SpiralNet() { register_module("l1", l1); register_module("l2", l2); }
//   Variable forward(const Variable& x) override {
//     // Use tanh nonlinearity which often helps on spiral-like problems
//     auto h = ag::tanhv(l1.forward(x));
//     return l2.forward(h);
//   }
// protected:
//   std::vector<ag::Variable*> _parameters() override { return {}; }
// };

// static void make_spiral_dataset(std::size_t points_per_class,
//                                 std::size_t classes,
//                                 std::vector<float>& X,
//                                 std::vector<std::size_t>& y) {
//   std::mt19937 rng(1337);
//   std::normal_distribution<float> noise(0.0f, 0.05f);

//   const std::size_t B = points_per_class * classes;
//   X.assign(B * 2, 0.0f);
//   y.assign(B, 0u);

//   for (std::size_t k = 0; k < classes; ++k) {
//     for (std::size_t i = 0; i < points_per_class; ++i) {
//       float r = static_cast<float>(i) / static_cast<float>(points_per_class - 1);
//       float t = 2.5f * r * 2.0f * float(M_PI) + (2.0f * float(M_PI) * k / classes);
//       float x1 = r * std::cos(t) + noise(rng);
//       float x2 = r * std::sin(t) + noise(rng);
//       std::size_t idx = k * points_per_class + i;
//       X[2 * idx + 0] = x1;
//       X[2 * idx + 1] = x2;
//       y[idx] = static_cast<std::size_t>(k);
//     }
//   }
// }

// TEST("nn/module_sgd_spiral_multiclass_converges") {
//   const std::size_t classes = 3;
//   const std::size_t n_per_class = 120; // total B = 360
//   std::vector<float> xv; std::vector<std::size_t> yv;
//   make_spiral_dataset(n_per_class, classes, xv, yv);
//   const std::size_t B = classes * n_per_class;

//   Variable X(xv, {B, 2}, /*requires_grad=*/true);

//   SpiralNet net;
//   // Tuned optimizer: moderate LR with momentum + Nesterov for stable, faster convergence
//   SGD opt(/*lr=*/0.5f, /*momentum=*/0.9f, /*nesterov=*/true, /*weight_decay=*/0.0f);

//   // Train on full batch with early stopping
//   float first_loss = 0.0f;
//   float final_loss = 0.0f;
//   int   max_steps = 4000; // increased from 1500
//   for (int step = 0; step < max_steps; ++step) {
//     auto logits = net.forward(X);
//     auto loss = ag::nn::cross_entropy(logits, yv);
//     if (step == 0) first_loss = loss.value()[0];
//     loss.backward();
//     opt.step(net);

//     // debug logging: print every 100 steps
//     if ((step % 100) == 0) {
//       float cur = loss.value()[0];
//       std::cout << "[spiral] step=" << step << " loss=" << cur << std::endl;
//       if (std::isnan(cur) || !std::isfinite(cur)) {
//         std::cerr << "[spiral] loss became NaN/Inf at step " << step << ", aborting." << std::endl;
//         break;
//       }
//     }

//     // check loss every 50 steps for early stop
//     if ((step % 50) == 49) {
//       auto logits_chk = net.forward(X);
//       auto loss_chk = ag::nn::cross_entropy(logits_chk, yv);
//       final_loss = loss_chk.value()[0];
//       // quick accuracy estimate to break early
//       const auto& zchk = logits_chk.value();
//       std::size_t correct = 0;
//       for (std::size_t i = 0; i < B; ++i) {
//         float z0 = zchk[3*i + 0], z1 = zchk[3*i + 1], z2 = zchk[3*i + 2];
//         std::size_t pred = 0; float best = z0;
//         if (z1 > best) { best = z1; pred = 1; }
//         if (z2 > best) { best = z2; pred = 2; }
//         if (pred == yv[i]) ++correct;
//       }
//       float acc = float(correct) / float(B);
//       if (final_loss <= first_loss * 0.3f && acc >= 0.9f) {
//         break;
//       }
//     }
//   }

//   // Final metrics
//   auto logitsF = net.forward(X);
//   auto lossF = ag::nn::cross_entropy(logitsF, yv);
//   final_loss = lossF.value()[0];
//   ASSERT_LE(final_loss, first_loss * 0.3f);

//   // Check accuracy
//   const auto& z = logitsF.value();
//   std::size_t correct = 0;
//   for (std::size_t i = 0; i < B; ++i) {
//     // argmax over 3 classes
//     float z0 = z[3*i + 0], z1 = z[3*i + 1], z2 = z[3*i + 2];
//     std::size_t pred = 0; float best = z0;
//     if (z1 > best) { best = z1; pred = 1; }
//     if (z2 > best) { best = z2; pred = 2; }
//     if (pred == yv[i]) ++correct;
//   }
//   float acc = float(correct) / float(B);
//   ASSERT_GE(acc, 0.9f);
// }
